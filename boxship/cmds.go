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
