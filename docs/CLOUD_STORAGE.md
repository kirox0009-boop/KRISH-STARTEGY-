# Moving storage off the VPS

A VPS disk is the scarcest thing on a box that also has to run MetaTrader, a
browser, and whatever else you need. This is how to keep KRISH's footprint near
zero without giving anything up.

There are three separate pieces of storage. They move independently, and you can
do one, two, or all three.

| Piece | Size | Where it can go | Effort |
|---|---|---|---|
| **Database** (experiments, verdicts) | grows — the main one | managed Postgres | one line in `.env` |
| **Price cache** (Parquet) | ~10 MB, bounded | S3-compatible object storage | env + one config line |
| **Delivered ZIPs** | ~40 KB each | S3-compatible object storage | same as above |

> **Before doing any of this:** check whether you need to. The `librarian` agent
> already caps growth — open the **Disk** panel in the control room. If the
> database is sitting at a few hundred MB and stable, you are fine. Do this when
> the numbers actually bother you, not on principle.

---

## 1. Database → free managed Postgres (biggest win, least work)

The blackboard is the only part that grows without bound. KRISH already speaks
Postgres, so this is a configuration change, not a code change.

**Free options:** [Neon](https://neon.tech) (recommended — generous free tier,
scale-to-zero), [Supabase](https://supabase.com), [Railway](https://railway.app).

### Steps

1. Create a free project on Neon. Choose the region closest to your VPS.
2. Copy the connection string it gives you. It looks like:

   ```
   postgresql://user:password@ep-something-123.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

3. Install the Postgres driver on the VPS:

   ```powershell
   cd C:\krish
   .venv\Scripts\python.exe -m pip install -e "backend[postgres]"
   ```

4. Put it in `C:\krish\.env`, changing the scheme to `postgresql+psycopg://`:

   ```ini
   DATABASE_URL=postgresql+psycopg://user:password@ep-something-123.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```

5. Restart KRISH. Tables are created automatically on first connect.

**Result:** the VPS holds no database at all. Growth becomes someone else's disk.

> ⚠️ The existing SQLite data is **not** migrated — KRISH starts with a fresh
> ledger. If you want to keep the history, say so and I will write a migration.
> Otherwise this is the cleanest moment to switch: early, before there is much
> to lose.

---

## 2. Price cache and ZIPs → object storage

**Recommended: [Cloudflare R2](https://developers.cloudflare.com/r2/)** — 10 GB
free, and crucially **no egress fees**, which matters because the factory
re-downloads cached data on demand. Backblaze B2, Wasabi, MinIO and AWS S3 all
work identically; only the endpoint URL differs.

### Steps (Cloudflare R2)

1. Cloudflare dashboard → **R2** → **Create bucket**, name it `krish`.
2. **Manage R2 API Tokens** → **Create API token** → permission
   *Object Read & Write* → scope it to that one bucket.
3. Copy the **Access Key ID**, **Secret Access Key**, and the
   **S3 endpoint** (`https://<account-id>.r2.cloudflarestorage.com`).
4. Install the extra on the VPS:

   ```powershell
   cd C:\krish
   .venv\Scripts\python.exe -m pip install -e "backend[cloud]"
   ```

5. Add to `C:\krish\.env`:

   ```ini
   S3_BUCKET=krish
   S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
   S3_ACCESS_KEY_ID=<your key id>
   S3_SECRET_ACCESS_KEY=<your secret>
   S3_REGION=auto
   ```

6. In `C:\krish\config\factory.yaml`:

   ```yaml
   storage:
     backend: s3
     offload_price_cache: true
     upload_packages: true
     keep_local_packages: false   # delete the local ZIP once it is uploaded
     local_cache_keep_days: 7     # prune local Parquet after a week, if uploaded
   ```

7. Restart KRISH.

### What then happens

- Every Parquet file is uploaded after it is written.
- The `librarian` deletes local Parquet older than `local_cache_keep_days` —
  **but only after confirming the object store actually has that exact file.**
  A missing re-download costs seconds; losing years of history to an eager
  cleanup would be much worse, so the check is not optional.
- On a cache miss, KRISH pulls the file back from the bucket before it considers
  re-fetching from the data provider.
- Delivered ZIPs are uploaded before anyone is told the package exists, so a
  recorded URL is always real.
- With `keep_local_packages: false`, the local ZIP is removed and the control
  room's **download** link redirects to object storage. If you did not set a
  public domain, the redirect is a **time-limited signed link**, so the bucket
  stays private — which is the right default for trading strategies.

Check it worked: the **Disk** panel and `GET /api/health/storage` both report the
active object store, bucket and prefix.

---

## Cost reality check

At a realistic few thousand strategies a day:

| | Monthly |
|---|---|
| Neon Postgres free tier | **$0** (0.5 GB storage — plenty; the librarian keeps it well under) |
| Cloudflare R2 free tier | **$0** (10 GB, no egress) |
| VPS disk used by KRISH | **well under 100 MB** — logs capped at 120 MB, cache pruned, DB remote |

You should not be paying anything for this. If you find yourself outgrowing both
free tiers, that means the factory is producing at a scale worth talking about,
and we should revisit retention settings first.

---

## Rolling back

Set `storage.backend: local` in `config/factory.yaml`, and remove
`DATABASE_URL` from `.env` to go back to SQLite. Nothing in the code path
changes — every remote feature is written to degrade to a no-op when
unconfigured, so a partly-configured setup keeps running rather than failing.
