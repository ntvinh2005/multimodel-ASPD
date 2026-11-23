# Browser demo from HiPerGator

This procedure serves the already-built research frontend from a time-limited CPU allocation. It
does not load model weights, run inference, request a GPU, or expose a public port.

## Boundaries

- Work from `/blue/<group>/<user>`, not `/home` or `/orange`.
- Use a CPU compute/development allocation. Do not keep Vite running on a login node.
- Use an SSH tunnel from the researcher's computer. Do not open the server to the public internet.
- Stop the allocation when the review is finished. The example below also applies a two-hour limit.
- The measured validation page uses the aggregate record committed with the frontend. The interactive
  component graph remains explicitly synthetic until a bounded raw-result export is available.

## 1. Prepare the checkout

Run these lightweight repository-management commands on a HiPerGator login shell:

```bash
cd /blue/<group>/<user>
git clone https://github.com/ntvinh2005/multimodel-ASPD.git
cd multimodel-ASPD
git switch main
git pull --ff-only origin main
```

If the frontend commit has not yet been merged, replace the last two commands with the integration
branch reported by the developer:

```bash
git fetch origin integration/universal-workbench-main
git switch --detach origin/integration/universal-workbench-main
```

## 2. Obtain a time-limited CPU shell

Use Open OnDemand to request a terminal on a CPU compute node for two hours, or use the group's
approved Slurm account/QOS. The exact account and QOS are group-specific. A representative command is:

```bash
srun --account=<group-account> --qos=<group-qos> \
  --time=02:00:00 --cpus-per-task=2 --mem=4G --pty bash -l
```

Do not add a GPU request. Once the shell starts, record the compute hostname:

```bash
hostname -f
```

## 3. Build and serve

Discover and load an available Node.js module; use Node 20 or newer:

```bash
module spider nodejs
module load nodejs/<available-version>
node --version
```

Then build from the frontend directory and serve it for at most two hours:

```bash
cd /blue/<group>/<user>/multimodel-ASPD/multimodel_dashboard
npm ci
npm test
npm run build
timeout --signal=INT 2h npm run dev -- --host 0.0.0.0 --port 5173
```

The expected terminal output includes `VITE ... ready`. Keep this compute shell open while reviewing.
The default workbench route does not require the optional Python API used by the older dashboard
screens.

## 4. Tunnel from the personal computer

On the researcher's computer, replace `<compute-hostname>` with the value printed by `hostname -f`:

```bash
ssh -N -L 5173:<compute-hostname>:5173 <username>@hpg.rc.ufl.edu
```

After MFA succeeds, open:

```text
http://127.0.0.1:5173/
```

Keep the SSH command running while using the frontend. The port is reachable only through this SSH
session.

## 5. Stop and verify cleanup

When review is complete, press `Ctrl+C` in the server shell, exit the compute shell, and close the SSH
tunnel. If the allocation was started through Open OnDemand, delete/end the interactive session there.
The `timeout` command stops Vite automatically after two hours, but it does not replace ending the
allocation explicitly.

## Agent handoff

An agent operating on HiPerGator may perform the repository checkout, CPU-session setup, dependency
installation, tests, build, and time-limited server startup above. It must report:

1. the exact Git commit;
2. the compute hostname;
3. the local tunnel command with placeholders, never credentials;
4. test and build results;
5. the automatic stop time;
6. whether the displayed graph is measured or synthetic.

The agent must not submit a GPU job, load a checkpoint, run model inference, expose a public listener,
or leave a server running on a login node.
