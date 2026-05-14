package main

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Config holds connection details. Password is read from env only — never
// passed on the command line so it doesn't end up in shell history or process
// listings.
type Config struct {
	Host      string
	Port      int
	User      string
	Password  string
	RemoteDir string // default working dir for deploy commands
}

func LoadConfig() (Config, error) {
	cfg := Config{
		Host:      getenv("BOXSHIP_HOST", ""),
		User:      getenv("BOXSHIP_USER", ""),
		Password:  os.Getenv("BOXSHIP_PASS"),
		RemoteDir: getenv("BOXSHIP_REMOTE_DIR", "~/box-vision"),
	}

	if v := os.Getenv("BOXSHIP_PORT"); v != "" {
		p, err := strconv.Atoi(v)
		if err != nil {
			return cfg, fmt.Errorf("BOXSHIP_PORT: %w", err)
		}
		cfg.Port = p
	}
	if cfg.Port == 0 {
		cfg.Port = 22
	}

	// Tilde expansion happens server-side. Just trim trailing slash for tidiness.
	cfg.RemoteDir = strings.TrimRight(cfg.RemoteDir, "/")

	var missing []string
	if cfg.Host == "" {
		missing = append(missing, "BOXSHIP_HOST")
	}
	if cfg.User == "" {
		missing = append(missing, "BOXSHIP_USER")
	}
	if cfg.Password == "" {
		missing = append(missing, "BOXSHIP_PASS")
	}
	if len(missing) > 0 {
		return cfg, errors.New("missing required env vars: " + strings.Join(missing, ", "))
	}
	return cfg, nil
}

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
