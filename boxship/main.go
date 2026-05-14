// boxship — tiny SSH/SFTP CLI for BoxVision remote training.
//
// Replaces sshpass + scp + ad-hoc shell scripts with a single static binary that
// reads credentials from env (never argv) and supports the deploy flows we need:
// inspect, run, upload, and a higher-level deploy.
//
//   BOXSHIP_HOST=10.0.0.103 BOXSHIP_USER=ubuntu BOXSHIP_PASS=00 \
//       boxship inspect
//
// Commands:
//
//   inspect            Run inventory on the remote (Python/GPU/CPU/disk/memory)
//   run <cmd...>       Execute a single command, stream output
//   upload <l> <r>     Copy a local file or directory to the remote (SFTP)
//
// Excludes from uploads by default: .venv, __pycache__, .git, runs/, etc.

package main

import (
	"fmt"
	"os"
	"strings"
)

const usage = `boxship — SSH/SFTP deploy tool

Usage:
  boxship inspect
  boxship run <command...>
  boxship run --pty <command...>      # allocate a PTY (nicer tqdm output)
  boxship upload <local> <remote> [--exclude=pat,pat]

Env:
  BOXSHIP_HOST       remote hostname/IP        (required)
  BOXSHIP_USER       remote username           (required)
  BOXSHIP_PASS       remote password           (required, env only)
  BOXSHIP_PORT       SSH port                  (default 22)
  BOXSHIP_REMOTE_DIR default remote workdir    (default ~/box-vision)
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	sub := os.Args[1]
	rest := os.Args[2:]

	cfg, err := LoadConfig()
	if err != nil {
		fmt.Fprintln(os.Stderr, "boxship: config error:", err)
		fmt.Fprintln(os.Stderr, "\n"+usage)
		os.Exit(2)
	}

	client, err := Dial(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "boxship: dial error:", err)
		os.Exit(1)
	}
	defer client.Close()

	switch sub {
	case "inspect":
		exit(cmdInspect(client))

	case "run":
		pty := false
		if len(rest) > 0 && rest[0] == "--pty" {
			pty = true
			rest = rest[1:]
		}
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "boxship: run requires a command")
			os.Exit(2)
		}
		exit(cmdRun(client, strings.Join(rest, " "), pty))

	case "upload":
		if len(rest) < 2 {
			fmt.Fprintln(os.Stderr, "boxship: upload requires <local> <remote>")
			os.Exit(2)
		}
		local, remote := rest[0], rest[1]
		var extra []string
		for _, arg := range rest[2:] {
			if strings.HasPrefix(arg, "--exclude=") {
				extra = strings.Split(strings.TrimPrefix(arg, "--exclude="), ",")
			}
		}
		exit(cmdUpload(client, local, remote, extra))

	case "-h", "--help", "help":
		fmt.Print(usage)

	default:
		fmt.Fprintf(os.Stderr, "boxship: unknown command %q\n\n%s", sub, usage)
		os.Exit(2)
	}
}

func exit(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "boxship:", err)
		os.Exit(1)
	}
}
