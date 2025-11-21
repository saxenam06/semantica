# Testing Documentation Locally

## Quick Start

### Install Dependencies

```bash
pip install -r requirements-docs.txt
```

### Run Local Server

```bash
mkdocs serve
```

Then visit: `http://127.0.0.1:8000`

### Build Static Site

```bash
mkdocs build
```

Output will be in the `site/` directory.

## Development Workflow

1. Make changes to `.md` files in `docs/`
2. Run `mkdocs serve` to preview
3. Check changes in browser
4. Commit and push when ready

## Troubleshooting

### Port Already in Use

```bash
mkdocs serve -a 127.0.0.1:8001
```

### Clear Cache

```bash
mkdocs build --clean
```

### Check Configuration

```bash
mkdocs build --verbose
```

