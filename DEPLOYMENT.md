# Deployment Guide

This guide covers deploying Dialectus Engine to production environments using environment variables for configuration.

## Table of Contents
- [Environment Variables](#environment-variables)
- [Production Configuration](#production-configuration)
- [Security Checklist](#security-checklist)
- [Local Development](#local-development)

---

## Environment Variables

All configuration can be overridden with environment variables. This is the **recommended approach for production** to keep secrets out of version control.

### Required for Production

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key for cloud models | `sk-or-v1-...` |
| `JWT_SECRET_KEY` | Secret key for signing JWT tokens | Generate with `openssl rand -hex 32` |
| `AUTH_DEVELOPMENT_MODE` | Disable for production email verification | `false` |
| `DATABASE_PATH` | Path to SQLite database (use persistent storage) | `/data/debates.db` |

### Email Service (Required if enabling email)

| Variable | Description | Example |
|----------|-------------|---------|
| `EMAIL_ENABLED` | Enable email sending | `true` |
| `SMTP_SERVER` | SMTP server hostname | `mail.example.com` |
| `SMTP_PORT` | SMTP port (587 for TLS, 465 for SSL) | `587` |
| `SMTP_USER` | SMTP username | `no-reply@yourdomain.com` |
| `SMTP_PASSWORD` | SMTP password | `your-password` |
| `SMTP_FROM_EMAIL` | From email address | `no-reply@yourdomain.com` |
| `SMTP_FROM_NAME` | From name in emails | `Your App Name` |
| `SMTP_USE_TLS` | Use TLS encryption | `true` |
| `FRONTEND_URL` | Frontend URL for email links | `https://yourdomain.com` |

### Web Server

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | HTTP server port | `8000` |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) | `https://yourdomain.com` |

### Optional Overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_EXPIRE_HOURS` | JWT token expiration | `168` (7 days) |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |

---

## Production Configuration

### 1. Database Persistence

**IMPORTANT**: Container/server deployments are often ephemeral - files not in persistent storage are lost on restart.

- Mount a persistent volume (e.g., `/data`)
- Set `DATABASE_PATH=/data/debates.db`
- Ensure the volume persists across deployments

### 2. Required Environment Variables

Set these in your deployment platform's environment configuration:

```bash
# Authentication & Security
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
JWT_SECRET_KEY=your-generated-secret-key-here
AUTH_DEVELOPMENT_MODE=false
DATABASE_PATH=/data/debates.db

# Email Service (if enabled)
EMAIL_ENABLED=true
SMTP_SERVER=mail.example.com
SMTP_PORT=587
SMTP_USER=no-reply@yourdomain.com
SMTP_PASSWORD=your-smtp-password-here
SMTP_FROM_EMAIL=no-reply@yourdomain.com
SMTP_FROM_NAME=Your App Name
SMTP_USE_TLS=true
FRONTEND_URL=https://yourdomain.com

# CORS
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 3. Generate Secure Secrets

```bash
# Generate JWT secret (Linux/macOS)
openssl rand -hex 32

# Or using Python
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Configuration File Behavior

On first run, if `debate_config.json` doesn't exist, it will be auto-created from `debate_config.example.json`.

**For production**: Environment variables override config file values, so the ephemeral config file doesn't matter - just ensure your environment variables are set correctly.

---

## Security Checklist

Before going to production, verify:

- [ ] `JWT_SECRET_KEY` is set to a strong random value (not the dev default)
- [ ] `OPENROUTER_API_KEY` is set via environment variable (not in code)
- [ ] `SMTP_PASSWORD` is set via environment variable (not in code)
- [ ] `AUTH_DEVELOPMENT_MODE=false` (requires email verification)
- [ ] `EMAIL_ENABLED=true` with valid SMTP configuration
- [ ] `DATABASE_PATH` points to persistent storage
- [ ] `ALLOWED_ORIGINS` is set to your frontend domain(s)
- [ ] HTTPS/TLS is enabled (handled by reverse proxy/platform)
- [ ] `debate_config.json` is in `.gitignore` (already configured)
- [ ] Never commit config files with secrets to version control

---

## Local Development

For local development, you don't need environment variables:

1. Copy `debate_config.example.json` to `debate_config.json` (automatic on first run)
2. Edit `debate_config.json` with your local settings
3. Add your OpenRouter API key to `debate_config.json` (this file is gitignored)
4. Set `auth.development_mode: true` to skip email verification
5. Run `python main.py --web`

`debate_config.json` is already in `.gitignore`, so your local secrets stay safe.

---

## Common Issues

### "Email service explicitly disabled in configuration"
- **Cause**: `EMAIL_ENABLED` environment variable not set
- **Fix**: Set `EMAIL_ENABLED=true` in your deployment environment

### "No OpenRouter API key found"
- **Cause**: `OPENROUTER_API_KEY` environment variable not set
- **Fix**: Add the environment variable in your deployment platform

### Database resets after deployment
- **Cause**: SQLite file is not in persistent storage
- **Fix**: Mount persistent volume and set `DATABASE_PATH` to point to it

### JWT "Invalid token" errors after redeployment
- **Cause**: `JWT_SECRET_KEY` changed or not set consistently
- **Fix**: Set `JWT_SECRET_KEY` environment variable to a fixed value

### CORS errors from frontend
- **Cause**: `ALLOWED_ORIGINS` not set or incorrect
- **Fix**: Set `ALLOWED_ORIGINS` to your frontend domain(s), comma-separated

---

## Environment Variable Priority

The system uses this priority order:

1. **Environment variables** (highest priority)
2. **debate_config.json** file
3. **Default values** in code

This means you can:
- Use config file for local development
- Use environment variables to override in production
- Mix both (env vars override config file values)