# ASPD v3 2B storage reclamation audit — 2026-11-01

This record precedes the researcher-authorized removal of two obsolete local project copies. It
does not authorize deletion of either GitHub repository or any ASPD artifact.

## `/blue/thai/vinhnguyen1/document-scanner`

- Resolved path: `/blue/thai/vinhnguyen1/document-scanner`.
- Remote: `https://github.com/connor-kress/document-scanner.git`.
- HEAD: `5dc032d6268462651a1a9165720623ef4c6ce016` on `feat/training`.
- Local `feat/training` and `main` both point to `5dc032d`; the commit is reachable from
  `origin/feat/training` after a fresh fetch.
- No detached or unreachable commit was reported. Four untracked files were present. They were
  preserved in
  `out/multimodel/aspd_v3/main_2b_v1/storage_cleanup_backups/document-scanner-uncommitted-20261101.tar.gz`
  with SHA-256 `1b44ff5c4b8e34a30f872cb0939ff65047f867017278f0be0ee68902fd79b637`.
- The local project used 5.0 GiB. Its ignored data, exports, results, environment and cache were
  explicitly authorized for removal by the researcher.

## `/blue/thai/vinhnguyen1/ia-unforeseen-70b`

- Resolved path: `/blue/thai/vinhnguyen1/ia-unforeseen-70b`.
- Remote: `https://github.com/ntvinh2005/ia-unforeseen-70b.git`.
- HEAD: `264c710edf6894d858e45bd40f49fd89aeea7a79` on `main`, equal to `origin/main` after a fresh fetch.
- Local `fix/audit-pipeline` at `7276cd743f12fe39e82694e9c3d35de4e01e7e8a` is reachable from
  `origin/fix/audit-pipeline` and `origin/main`.
- No detached or unreachable commit was reported. Two modified files and one untracked test were
  present. They were preserved in
  `out/multimodel/aspd_v3/main_2b_v1/storage_cleanup_backups/ia-unforeseen-70b-uncommitted-20261101.tar.gz`
  with SHA-256 `80f80ccd1739daa69cd956239f52c342e571458abb8386e58f7222a15f30457b`.
- Large ignored contents included 132 GiB under `models/Llama-3.3-70B-Instruct` and 65 GiB under
  `outputs`; the prior audit measured approximately 452 GB logical project content after its first
  cache cleanup. The researcher explicitly authorized removal of the project.

## Dependency and deletion decision

- `git worktree list` reported only each project's own root.
- Slurm had no active job for the account, and no process other than this audit's read-only `du`
  scan referenced either path.
- Neither path contains a nested mount. Both are physical directories owned by the current user.
- Backup archives passed `sha256sum -c` and listed all seven dirty/untracked source files.
- Decision: delete exactly these two local directory trees, then measure actual Blue quota recovery.

## Result

- Both exact paths were removed. Neither GitHub repository was modified or deleted.
- Blue group usage changed from 4.573 TB to 4.157 TB after Lustre quota accounting settled,
  recovering approximately 0.416 TB. User-attributed usage changed from 1.851 TB to 1.435 TB.
- The verified backup archives occupy 1.1 MiB and remain under the isolated ASPD v3 2B output
  namespace.
- Blue group headroom after cleanup is approximately 0.843 TB against the 5 TB hard limit.

