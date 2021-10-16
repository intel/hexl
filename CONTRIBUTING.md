# Pull Requests

Intel HEXL welcomes pull requests from external contributors to the `main` branch.

Before contributing, please run
```bash
cmake --build build --target check unittest
```
to make sure the formatting checks and all unit tests pass.

Please sign your commits before making a pull request. See instructions [here](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/signing-commits) for how to sign commits.

### Known Issues ###

* ```Executable `cpplint` not found```

  Make sure you install cpplint: ```pip install cpplint```.
  If you install `cpplint` locally, make sure to add it to your `PATH`.

* ```/bin/sh: 1: pre-commit: not found```

  Install `pre-commit`. More info at https://pre-commit.com/.

* ```
     error: gpg failed to sign the data
     fatal: failed to write commit object
  ```
  Try adding ```export GPG_TTY=$(tty)``` to `~/.bashrc`.
