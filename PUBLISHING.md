# Publishing

`syntri-contracts` publishes to PyPI automatically when a version tag is pushed.
The `publish.yml` workflow uses PyPI trusted publishing (OIDC) — there is no API
token stored in GitHub secrets.

## Before the first publish

1. Create the `syntri-contracts` project on PyPI.
2. Add a trusted publisher under the project's Publishing settings:
   - Owner: `DomaniLabs`
   - Repository: `Syntri`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. Create the `pypi` environment on GitHub (Settings → Environments).

## Then tag and push

```bash
git tag -a v0.3.1 -m "syntri-contracts 0.3.1"
git push origin --tags
```

Pushing a `v*` tag triggers the publish workflow, which builds the sdist and
wheel and uploads them to PyPI. Do not push a tag until the trusted publisher
and the `pypi` environment are configured, or the publish step will fail.
