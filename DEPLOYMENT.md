# Hosting and performance

## Try the optimized Streamlit deployment first

1. In Streamlit Community Cloud, open **Manage app** for SkinGuardAI.
2. Verify the repository is `NithishMee/SkinGuardAI`, branch `main`, entrypoint `app.py`.
3. Wait for the GitHub update to deploy. If it has not refreshed, use **Reboot app**.
4. Inspect build logs. `uv.lock` now selects CPU PyTorch and excludes optional training tools from the default installation. Streamlit prioritizes this lockfile over `requirements.txt`.
5. Open the page in a fresh browser session. Enter details and upload a dermoscopy image. The model loads on the first **Analyze image** request; later analyses reuse it.
6. Select **Generate image attribution** only when needed. Use **Prepare reports** to generate downloads. After changing notes or the heatmap palette, prepare the reports again.

This is a reasonable free demonstration host. Hosting changes cannot remove the model's CPU cost, and the first analysis after a process restart still initializes the model.

## Render: a paid, always-on option

For regular use, start by evaluating **1 CPU / 2 GB RAM (`1c-2g`)**, and measure memory/latency under your expected traffic. This is a starting point, not a capacity guarantee. Consider **2 CPU / 4 GB (`2c-4g`)** for more headroom. Review current charges before creating the service. No paid service is provisioned by this repository update.

1. Sign in to [Render](https://dashboard.render.com/) with GitHub.
2. Choose **New → Web Service**, connect GitHub, and select `NithishMee/SkinGuardAI`.
3. Select branch `main`, runtime **Python**, and leave the root directory empty.
4. Choose the region closest to your visitors (for visitors in India, consider Singapore).
5. Set the build command:
   ```sh
   pip install uv==0.11.29 && uv sync --frozen --no-dev
   ```
6. Set the start command:
   ```sh
   .venv/bin/streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
7. Add environment variables `PYTHON_VERSION=3.12.10` and `SKINGUARD_CPU_THREADS=1` (try `2` on a two-CPU plan and benchmark).
8. Set the health check path to `/_stcore/health`.
9. Choose the paid compute plan, review the cost, and click **Deploy Web Service**.
10. Open the assigned `https://...onrender.com` URL. Verify intake, analysis, optional attribution, PDF/text downloads, and returning to intake. Test first and repeat analyses separately.
11. Watch CPU/RAM metrics and logs while testing with multiple visitors. Requests using the shared model serialize to prevent hook/attribution corruption; sustained concurrent demand needs additional capacity or a separate inference service.
12. Enable automatic deployments from `main` if desired. Share the new URL after testing; keep the existing Streamlit URL until then.

The repository's `render.yaml` has the same commands and health check but retains the existing **free** plan to avoid silently changing billing. If deploying it as a Blueprint, explicitly review/change that plan. Free Render services have 512 MB RAM / 0.1 CPU and spin down after 15 idle minutes; they are a poor fit for fixing this app's responsiveness.

## Local validation and training

### Measured development result

On the development Windows CPU environment (PyTorch 2.14.0, two Torch threads), five warm runs on a fixed synthetic 512 x 512 RGB image gave median times of **0.981 s** for the original classification plus automatic heatmap and **0.269 s** for classification alone (**3.65x** faster to the prediction). Class probabilities matched within `1e-6`. Optional heatmap work is deferred, not eliminated. These timings exclude network, cold imports, PDF generation, and cloud contention; they are not production latency or clinical accuracy measurements.

```sh
uv sync --frozen
uv run python -m unittest discover -s tests -v
uv run streamlit run app.py
```

Training dependencies are optional: `uv sync --extra train`. The lockfile targets CPU deployment. For GPU training, create a separate environment and install the appropriate PyTorch CUDA build with `requirements-train.txt`.

Upload previews are cached only in the visitor's session, not a shared data cache. Uploads are limited to 10 MB and 20 megapixels. The color-based input check is a heuristic, not a validated medical image detector. Model scores and heatmaps do not establish a diagnosis. The changes optimize execution, not model accuracy.

## References (checked October 23, 2026)

- [Streamlit dependency priority](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Render compute plans](https://render.com/docs/compute-plans)
- [Render free service limitations](https://render.com/docs/free)
- [Render pricing](https://render.com/pricing)
- [Render Blueprint reference](https://render.com/docs/blueprint-spec)
