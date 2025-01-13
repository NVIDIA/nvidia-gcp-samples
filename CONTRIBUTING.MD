# Contribute to the NVIDIA GCP Samples Project

## Contributions
NVIDIA is open to accept new samples of NVIDIA software tools application on GCP platforms. The NVIDIA GCP Samples Project is open-source and permissively licensed under the Apache 2.0 license with requirements for source code [contributions](#signing).

Please work with the NVIDIA team and GitHub repository maintainers to discuss questions and required testing for new contributions.

## Pull Request (PR) Guidelines
Please follow the process below for code contributions.

1. First [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/NVIDIA/nvidia-gcp-samples) NIM GCP Samples repository.

2. Git clone the forked repository and push changes to the personal fork.

  ```bash
git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git nvidia-gcp-samples
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote).
git push -u origin <local-branch>:<remote-branch>
  ```

3. Once the code changes are staged on the fork and ready for review, a [Pull Request](https://help.github.com/en/articles/about-pull-requests) (PR) can be [requested](https://help.github.com/en/articles/creating-a-pull-request) to merge the changes from a branch of the fork into a selected branch of upstream.
  * Exercise caution when selecting the source and target branches for the PR. 
    `master` branch should be the latest version of the upstream repo.
  * Creation of a PR creation kicks off the code review process.
  * Contact repository maintainer for the review.
  * While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. After completing appropriate testing, the PR will be reviewed by repository maintainer. If meet requirements, the PR will be approved and merged by the developer and/or maintainer.


## <a name="signing"></a>Signing your work

We require that all contributions are certified under the terms of the Developer Certificate of Origin (DCO), Version 1.1.

This certifies that the contribution is your original work or you have the right to submit it under the same or compatible license. Any public contribution that contains commits that are not signed off will not be accepted.

Both SSH and GPG-signed commits are accepted to fulfill the requirements of the DCO.

- To sign off on a commit, add the `--signoff` (or `-S`) flag when committing the changes:
   ```bash
   git commit -S -m "Your commit message"
   ```
   This will append the following to your commit message:
   ```
   Signed-off-by: Your Name <your@email.com>
   ```

- If you are using GPG key, follow the [instruction](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key) to generate a GPG key. Enabling signing by default is available if configure Git client with passphrase. 

   Set your primary GPG signing key in Git:
   ```bash
   git config --global user.signingkey YOUR_GPG_KEY_ID
   ```
   Enable signing the commit by default with GPG key:
   ```bash
   git config commit.gpgsign true
   ```
   (Optional) Enable signing all commits by default in any local repository with GPG key:
   ```bash
   git config --global commit.gpgsign true
   ```

If you have any questions with signing the commit, please refer to the [GitHub documentation](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).


### Developer Certificate of Origin (DCO)

To ensure the quality and legality of the code base, all contributors are required to certify the origin of their contributions under the terms of the Developer Certificate of Origin (DCO), Version 1.1:

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
  1 Letterman Drive
  Suite D4700
  San Francisco, CA, 94129

  Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

  (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
