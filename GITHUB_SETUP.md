# GitHub Repository Setup Checklist

This document provides step-by-step instructions for publishing your quantum reservoir computing project to GitHub.

## Prerequisites

- [ ] GitHub account created
- [ ] Git configured on your local machine (name and email)
- [ ] SSH key set up for GitHub (optional, but recommended) or HTTPS access configured

## Step 1: Update Placeholder Information

Before pushing to GitHub, update the following files with your actual information:

### LICENSE
- [ ] Replace `[Your Name]` in `LICENSE` with your actual name

### README.md
- [x] Replace `yourusername` in repository URLs with your GitHub username (already done: maxjeanfavre)
- [ ] Update citation section with your name and paper details (if applicable)

### pyproject.toml
- [ ] Replace `[Your Name]` with your actual name
- [ ] Replace `your.email@example.com` with your actual email

## Step 2: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `quantum_reservoir_compting` (or your preferred name)
   - **Description**: "A Python implementation of quantum reservoir computing for classification tasks"
   - **Visibility**: Choose Public (for open-source) or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 3: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Use these commands:

```bash
# Make sure you're in the project directory
cd /mnt/c/Users/maxim/quantum_reservoir_compting

# Add all files to git (if not already done)
git add .

# Create initial commit
git commit -m "Initial commit: Quantum Reservoir Computing implementation"

# Add GitHub remote
git remote add origin https://github.com/maxjeanfavre/quantum_reservoir_compting.git

# Or if using SSH:
# git remote add origin git@github.com:maxjeanfavre/quantum_reservoir_compting.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify Repository

- [ ] Visit your repository on GitHub
- [ ] Verify all files are present
- [ ] Check that README displays correctly
- [ ] Verify LICENSE file is visible
- [ ] Test cloning the repository in a new location to ensure it works

## Step 5: Optional Enhancements

### Add Repository Topics
On GitHub, click the gear icon next to "About" and add topics:
- `quantum-computing`
- `reservoir-computing`
- `machine-learning`
- `python`
- `qutip`
- `classification`

### Add Repository Description
Update the repository description on GitHub:
"A Python implementation of quantum reservoir computing for classification tasks. Implements quantum reservoirs using steady-state and dynamical density matrices, with comparisons to classical Echo State Networks (ESN)."

### Create a Release (Optional)
If you want to tag a specific version:
```bash
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0
```

## Troubleshooting

### Authentication Issues
If you encounter authentication errors:
- For HTTPS: Use a personal access token instead of password
- For SSH: Ensure your SSH key is added to GitHub account

### Large Files
If you have large files that shouldn't be in the repository:
- Check `.gitignore` is working correctly
- Use `git rm --cached <file>` to remove tracked files that should be ignored

### Updating Remote URL
If you need to change the remote URL:
```bash
git remote set-url origin https://github.com/maxjeanfavre/quantum_reservoir_compting.git
```

## Next Steps After Publishing

1. **Share your repository**: Share the link with collaborators or on academic platforms
2. **Add badges** (optional): Consider adding badges to README for build status, license, etc.
3. **Create issues**: Use GitHub Issues to track bugs and feature requests
4. **Write documentation**: Consider adding more detailed documentation in a `docs/` folder
5. **Add examples**: Create example notebooks or scripts in an `examples/` directory

## Repository Maintenance

- Keep dependencies up to date
- Update README as the project evolves
- Tag releases for important milestones
- Consider adding a CHANGELOG.md for version history

