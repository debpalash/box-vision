package main

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Default exclude patterns when uploading the project tree. Match either by
// basename or path prefix relative to the upload root.
var defaultExcludes = []string{
	".venv", "venv", "env",
	"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
	".git",
	"runs", "wandb", "mlruns", "tensorboard_logs",
	"node_modules", ".DS_Store",
	"boxship/boxship", // our own compiled binary
}

func cmdInspect(client *Client) error {
	// One short script that reports CPU/GPU/Python/disk/memory in one round-trip.
	script := `set -e
echo "=== UNAME ==="
uname -a
echo "=== PYTHON ==="
python3 --version 2>/dev/null || echo "(python3 not found)"
which uv || echo "(uv not installed)"
echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "(no nvidia-smi — CPU-only)"
fi
echo "=== CPU ==="
nproc
lscpu | head -5 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null
echo "=== MEMORY ==="
free -h | head -2
echo "=== DISK ==="
df -h ~ | tail -1
`
	_, err := client.Run(script, false)
	return err
}

func cmdRun(client *Client, command string, pty bool) error {
	code, err := client.Run(command, pty)
	if err != nil {
		return err
	}
	if code != 0 {
		return fmt.Errorf("remote command exited %d", code)
	}
	return nil
}

// cmdWait blocks until a marker string appears in a remote file, then prints
// a tail of that file and exits. Used to bridge "remote process finished" to
// the local harness's task-completion notification — run it as a background
// task and you get a ping when training is done.
//
// Implementation: a tiny shell loop on the remote that greps the file every
// pollSec seconds. Exits 0 on match, 1 on timeout, 2 if file is missing too
// long.
func cmdWait(client *Client, logPath, marker string, pollSec, timeoutSec int, tailLines int) error {
	if marker == "" {
		marker = "Training complete"
	}
	if pollSec <= 0 {
		pollSec = 30
	}
	if tailLines <= 0 {
		tailLines = 40
	}
	// Resolve leading ~/ remotely (the local shell would otherwise expand it
	// against the local home, pointing at the wrong path).
	if strings.HasPrefix(logPath, "~/") {
		logPath = "$HOME/" + logPath[2:]
	} else if logPath == "~" {
		logPath = "$HOME"
	}
	timeoutClause := ""
	if timeoutSec > 0 {
		timeoutClause = fmt.Sprintf(`if [ $waited -ge %d ]; then echo "boxship-wait: timeout after %ds" >&2; exit 1; fi`, timeoutSec, timeoutSec)
	}
	// Note: we expand `log` via eval below so $HOME interpolates remotely.
	script := fmt.Sprintf(`
set -u
log=$(eval echo %q)
marker=%q
poll=%d
waited=0
missing=0
echo "boxship-wait: watching $log for $marker (poll=${poll}s)"
while :; do
    if [ -f "$log" ]; then
        missing=0
        if grep -q -- "$marker" "$log"; then
            echo "boxship-wait: marker hit at $(date)"
            tail -%d "$log" | tr -d '\r'
            exit 0
        fi
    else
        missing=$((missing + poll))
        if [ $missing -ge 300 ]; then
            echo "boxship-wait: log not found for 5+ minutes: $log" >&2
            exit 2
        fi
    fi
    sleep $poll
    waited=$((waited + poll))
    %s
done
`, logPath, marker, pollSec, tailLines, timeoutClause)

	code, err := client.Run(script, false)
	if err != nil {
		return err
	}
	if code != 0 {
		return fmt.Errorf("remote wait exited %d", code)
	}
	return nil
}

// cmdUpload copies a local file or directory to the remote.
//
// extraExcludes is matched against either basename or path-relative-to-root.
func cmdUpload(client *Client, local, remote string, extraExcludes []string) error {
	info, err := os.Stat(local)
	if err != nil {
		return err
	}
	excludes := append([]string{}, defaultExcludes...)
	excludes = append(excludes, extraExcludes...)

	if !info.IsDir() {
		n, err := client.Upload(local, remote)
		if err != nil {
			return err
		}
		fmt.Printf("uploaded %s → %s (%.2f MB)\n", local, remote, float64(n)/1024/1024)
		return nil
	}

	root := filepath.Clean(local)
	if err := client.MkdirAll(remote); err != nil {
		return err
	}

	var nFiles int
	var nBytes int64
	err = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		if rel == "." {
			return nil
		}
		if isExcluded(rel, d.Name(), excludes) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		remotePath := filepath.ToSlash(filepath.Join(remote, rel))
		if d.IsDir() {
			return client.MkdirAll(remotePath)
		}
		n, uerr := client.Upload(path, remotePath)
		if uerr != nil {
			return fmt.Errorf("upload %s: %w", rel, uerr)
		}
		nFiles++
		nBytes += n
		fmt.Printf("  %s  (%.1f KB)\n", rel, float64(n)/1024)
		return nil
	})
	if err != nil {
		return err
	}
	fmt.Printf("uploaded %d files, %.2f MB → %s\n", nFiles, float64(nBytes)/1024/1024, remote)
	return nil
}

func isExcluded(relPath, basename string, patterns []string) bool {
	relPath = filepath.ToSlash(relPath)
	for _, p := range patterns {
		if p == basename {
			return true
		}
		if p == relPath {
			return true
		}
		if strings.HasPrefix(relPath, p+"/") {
			return true
		}
		// Suffix glob: "*.zip" style
		if strings.HasPrefix(p, "*") && strings.HasSuffix(basename, p[1:]) {
			return true
		}
	}
	return false
}
