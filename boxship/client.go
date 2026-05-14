// Package main: boxship — SSH/SFTP client tailored to BoxVision deploys.
//
// This file owns the connection lifecycle: SSH config (password auth), known-hosts
// handling, command streaming, and SFTP file transfer. Higher-level commands live
// in cmds.go.

package main

import (
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

type Client struct {
	cfg  Config
	ssh  *ssh.Client
	sftp *sftp.Client
}

// Dial opens an SSH connection using password auth, validating the host key
// against ~/.ssh/known_hosts (auto-adding on first contact).
func Dial(cfg Config) (*Client, error) {
	hostKeyCb, err := hostKeyCallback()
	if err != nil {
		return nil, fmt.Errorf("known_hosts: %w", err)
	}
	sshCfg := &ssh.ClientConfig{
		User:            cfg.User,
		Auth:            []ssh.AuthMethod{ssh.Password(cfg.Password)},
		HostKeyCallback: hostKeyCb,
		Timeout:         15 * time.Second,
	}
	addr := net.JoinHostPort(cfg.Host, fmt.Sprint(cfg.Port))
	conn, err := ssh.Dial("tcp", addr, sshCfg)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", addr, err)
	}
	return &Client{cfg: cfg, ssh: conn}, nil
}

// Run executes a command on the remote, streaming stdout+stderr to the local
// terminal. Returns the remote exit code.
func (c *Client) Run(cmd string, pty bool) (int, error) {
	sess, err := c.ssh.NewSession()
	if err != nil {
		return -1, err
	}
	defer sess.Close()

	if pty {
		modes := ssh.TerminalModes{
			ssh.ECHO:          0,
			ssh.TTY_OP_ISPEED: 14400,
			ssh.TTY_OP_OSPEED: 14400,
		}
		if err := sess.RequestPty("xterm-256color", 40, 120, modes); err != nil {
			return -1, fmt.Errorf("pty: %w", err)
		}
	}

	stdout, err := sess.StdoutPipe()
	if err != nil {
		return -1, err
	}
	stderr, err := sess.StderrPipe()
	if err != nil {
		return -1, err
	}

	if err := sess.Start(cmd); err != nil {
		return -1, err
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go pump(&wg, stdout, os.Stdout)
	go pump(&wg, stderr, os.Stderr)
	wg.Wait()

	if err := sess.Wait(); err != nil {
		if exitErr, ok := err.(*ssh.ExitError); ok {
			return exitErr.ExitStatus(), nil
		}
		return -1, err
	}
	return 0, nil
}

func pump(wg *sync.WaitGroup, src io.Reader, dst io.Writer) {
	defer wg.Done()
	_, _ = io.Copy(dst, src)
}

// openSFTP lazily opens an SFTP subsystem on the existing SSH connection.
func (c *Client) openSFTP() (*sftp.Client, error) {
	if c.sftp != nil {
		return c.sftp, nil
	}
	s, err := sftp.NewClient(c.ssh)
	if err != nil {
		return nil, err
	}
	c.sftp = s
	return s, nil
}

// MkdirAll creates a remote directory tree.
func (c *Client) MkdirAll(path string) error {
	s, err := c.openSFTP()
	if err != nil {
		return err
	}
	return s.MkdirAll(path)
}

// Upload copies a single local file to a remote path. Parent dirs are created.
func (c *Client) Upload(localPath, remotePath string) (int64, error) {
	s, err := c.openSFTP()
	if err != nil {
		return 0, err
	}
	if err := s.MkdirAll(filepath.Dir(remotePath)); err != nil {
		return 0, fmt.Errorf("mkdir %s: %w", filepath.Dir(remotePath), err)
	}
	src, err := os.Open(localPath)
	if err != nil {
		return 0, err
	}
	defer src.Close()
	dst, err := s.Create(remotePath)
	if err != nil {
		return 0, err
	}
	defer dst.Close()
	return io.Copy(dst, src)
}

// Close releases SFTP + SSH resources.
func (c *Client) Close() {
	if c.sftp != nil {
		_ = c.sftp.Close()
	}
	if c.ssh != nil {
		_ = c.ssh.Close()
	}
}

func hostKeyCallback() (ssh.HostKeyCallback, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	khPath := filepath.Join(home, ".ssh", "known_hosts")
	// Ensure the file exists so knownhosts.New doesn't error out on first contact.
	if _, err := os.Stat(khPath); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(khPath), 0o700); err != nil {
			return nil, err
		}
		f, err := os.OpenFile(khPath, os.O_CREATE|os.O_WRONLY, 0o600)
		if err != nil {
			return nil, err
		}
		_ = f.Close()
	}
	khChecker, err := knownhosts.New(khPath)
	if err != nil {
		return nil, err
	}
	// TOFU: if the host is unknown, append its key and accept. If known with a
	// different key, refuse (man-in-the-middle protection).
	return func(hostname string, remote net.Addr, key ssh.PublicKey) error {
		err := khChecker(hostname, remote, key)
		if err == nil {
			return nil
		}
		if kErr, ok := err.(*knownhosts.KeyError); ok && len(kErr.Want) == 0 {
			f, ferr := os.OpenFile(khPath, os.O_APPEND|os.O_WRONLY, 0o600)
			if ferr != nil {
				return ferr
			}
			defer f.Close()
			line := knownhosts.Line([]string{hostname, remote.String()}, key)
			fmt.Fprintln(os.Stderr, "boxship: trusting new host key for", hostname)
			_, werr := fmt.Fprintln(f, line)
			return werr
		}
		return err
	}, nil
}
