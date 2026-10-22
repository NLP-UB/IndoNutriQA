# RD / LLM-Judge Calibration Study — Setup and Workflow

Human-anchored validation of the LLM-as-judge rubric used in
`src/pipelines/14-semantic_evaluation.py`. The paper currently reports only
*judge-vs-judge* agreement (GPT-4o-mini vs Claude Haiku 4.5, pooled
κ = 0.652); this study adds the *judge-vs-Registered-Dietitian* arm and the
*inter-RD* arm, both stratified by question category.

Everything here is driven by `src/pipelines/18-rd_judge_agreement.py`.

---

## 0. What gets measured

| Comparison | Estimand | Why it matters |
|---|---|---|
| RD1 vs RD2 | Inter-RD κ | The ceiling. A judge cannot be expected to agree with RDs more than RDs agree with each other. |
| Judge vs RD consensus | Judge validity κ | The headline number the Limitations section promises. |
| Judge − RD mean score | Leniency bias | κ can be decent while the judge is systematically 0.5 points generous. Reported separately. |

All three are computed **pooled and per question category** (factual /
inferential / recommendation), because the working hypothesis is that
recommendation-style questions — where the NCP/PES rubric constrains the answer
least — are where the judge diverges most.

---

## 1. Draw the sample (no API calls)

```bash
python src/pipelines/18-rd_judge_agreement.py sample --n 50
```

Writes into `data/18-rd_judge_agreement/`:

| File | Purpose | Give to RDs? |
|---|---|---|
| `annotation_sample.json` | Full record incl. system name, tier, cached judge scores | **No** — unblinding |
| `label_studio_tasks.json` | Blinded tasks (unit id, diagnosis, anthropometry, question, reference, candidate) | Yes, via Label Studio |
| `scoring_sheet.csv` | Same fields as an offline spreadsheet | Only if not using Label Studio |

**Sampling design.** 50 units allocated evenly over 3 question categories
(17/17/16), and within each category evenly over 3 system quality tiers
(low/mid/high, cut at tertiles of mean token-F1 across the 22 prediction
files). Each unit is a distinct test item; the current draw covers all 20
source reports in the test split. Units are only proposed if their scores are
**already cached** for every judge under study, so the judge arm costs nothing.

Stratification is on question category and *aggregate* system quality —
never on the per-unit judge score. Conditioning on the judge score would bias
the very quantity being estimated.

Presentation order is shuffled so tier does not correlate with position in the
annotation queue.

### What the RD sees

Each task shows the unit id, the **medical diagnosis**, the **anthropometric
assessment**, the question, the reference answer, and the candidate answer.
The full source chunk is deliberately **not** shown — it is long, and the
structured metadata is what an RD actually needs to judge a nutrition answer.

The LLM judge sees only `(reference, candidate)`, so the RD has strictly more
information than the judge did. Say so in the paper: it can move judge-vs-RD κ
for reasons unrelated to judge quality.

### Strict-parity variant

To give the RD *exactly* the judge's information — the cleanest κ
interpretation — drop the metadata too:

```bash
python src/pipelines/18-rd_judge_agreement.py sample --n 50 --no-context
```

and delete the two marked metadata `<Text>` lines from `labeling_config.xml`.
Use the same variant for both RDs and state which one in the paper.

---

## 2. Install Label Studio

> Already running an instance? Skip to §3. Use whatever host port it is
> published on — the examples below assume `6003`, matching a container
> started with `-p 6003:8080`. `docker ps` shows the mapping.

### Option A — Docker (recommended for a server)

```bash
cd docs/rd_annotation
docker compose up -d
docker compose logs -f label-studio     # wait for "Starting server"
```

Open `http://<server>:8080`. The first account you create owns the
organisation. See `docker-compose.yml` for Postgres, HTTPS host, and volume
notes. **Back up `docs/rd_annotation/ls-data/`** — it holds the RD labels.

### Option B — pip

```bash
python -m venv .venv-ls && source .venv-ls/bin/activate
pip install label-studio
label-studio start --port 8080 --data-dir ~/label-studio-data
```

Label Studio pulls in a large dependency tree (Django, etc.) that conflicts
with this project's `requirements.txt` in places — keep it in its own
virtualenv, not the one running the pipelines.

To keep it alive across logouts on a server:

```bash
sudo tee /etc/systemd/system/label-studio.service >/dev/null <<'EOF'
[Unit]
Description=Label Studio (IndoNutriQA RD annotation)
After=network.target

[Service]
User=YOUR_USER
Environment="LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
ExecStart=/full/path/.venv-ls/bin/label-studio start --port 8080 --data-dir /home/YOUR_USER/label-studio-data
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now label-studio
```

---

## 3. Create the projects

> **Community edition constraint.** Per-project membership, annotator roles,
> and task distribution to named annotators are Label Studio **Enterprise**
> features. In Community, every member of the organisation can open every
> project and can see every annotation in the Data Manager — including the
> other RD's. Community cannot enforce blinding.
>
> The workaround is **one project per RD**, each loaded with the same 50 tasks.
> Each RD works only in their own project, so neither sees the other's labels
> in their own queue or Data Manager. This is operational blinding, not
> enforced blinding: state in the paper that RDs were instructed not to open
> each other's project. It also makes rater attribution trivial — one export
> per RD, merged by `import` with explicit labels.

### Create project 1 (repeat for project 2)

1. **Create Project** → name it `IndoNutriQA RD1` (then `IndoNutriQA RD2`).
2. **Labeling Setup → Custom template** → replace the whole box with the
   contents of `docs/rd_annotation/labeling_config.xml` → **Save**.
3. **Data Import** → upload
   `data/18-rd_judge_agreement/label_studio_tasks.json`, choosing **JSON** as
   the format. Confirm the project shows **50 tasks**.

Or via the API, which is easier to repeat identically for both projects:

```bash
LS_URL=http://<your-server>:6003
LS_TOKEN=<Account & Settings → Access Token>

for RD in RD1 RD2; do
  PID=$(curl -s -X POST "$LS_URL/api/projects/" \
    -H "Authorization: Token $LS_TOKEN" -H "Content-Type: application/json" \
    -d "{\"title\": \"IndoNutriQA $RD\",
         \"label_config\": $(python3 -c 'import json,sys; print(json.dumps(open("docs/rd_annotation/labeling_config.xml").read()))')}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
  echo "$RD -> project $PID"
  curl -s -X POST "$LS_URL/api/projects/$PID/import" \
    -H "Authorization: Token $LS_TOKEN" -H "Content-Type: application/json" \
    --data-binary @data/18-rd_judge_agreement/label_studio_tasks.json
done
```

### Settings that matter, per project

Project **Settings → Annotation**:

| Setting | Value | Why |
|---|---|---|
| Show "Skip" button | **Off** | A skipped unit is a missing cell in both κ matrices. |
| Allow empty annotations | **Off** | Forces all three dimensions to be scored. |
| Reveal pre-annotations interactively | **Off** | No judge score can leak. |

Paste `docs/rd_annotation/rubric_short.html` (everything after the comment
block) into **Labeling Instructions** on the same tab and tick **Show before
labeling**. It is the one-screen reminder — each RD still reads the full
`rubric_id.md` once before starting.

Leave **Annotations per task / overlap at 1** in the one-project-per-RD layout
— each project has exactly one annotator.

### Alternative: single project, overlap 2

If you would rather run one project, set `maximum_annotations` to 2 and let
both RDs pull from the same queue:

```bash
curl -s -X PATCH "$LS_URL/api/projects/$PID/" \
  -H "Authorization: Token $LS_TOKEN" -H "Content-Type: application/json" \
  -d '{"maximum_annotations": 2}'
```

This works in Community and gives a single export that `import` splits by
`completed_by`. The cost is blinding: both RDs can see each other's
annotations in the Data Manager. Prefer the two-project layout.

## 3b. Give the RDs access

Community has one organisation, and every member sees every project — there is
no per-project invite.

1. **Organisation** (top-right menu → *Organization*) → **Add People**.
2. Copy the invite link and send it to each RD privately. Anyone with that link
   can join the organisation, so do not post it in a group chat.
3. Each RD signs up with their own email — this is what makes `completed_by`
   distinguishable if you ever fall back to the single-project layout.
4. Send each RD **only their own project URL**
   (`http://<your-server>:6003/projects/<id>/`) plus `rubric_id.md`, and tell
   them explicitly not to open the other project.
5. Confirm each RD sees 50 tasks, then have them click **Label All Tasks**.

If the server is reachable from the public internet, set
`LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK: "true"` in `docker-compose.yml` and
restart, so only invite-link holders can create accounts.

---

## 4. Calibration session

Before the real 50, run the 5-unit calibration described in §7 of
`rubric_id.md`. Draw calibration units from a different seed so they are not
part of the study sample:

```bash
python src/pipelines/18-rd_judge_agreement.py sample --n 5 --seed 7
# move the outputs aside, import as a SEPARATE Label Studio project,
# then re-run the real draw:
python src/pipelines/18-rd_judge_agreement.py sample --n 50
```

Discuss disagreements on those 5 with both RDs, then start the real task. **No
discussion between RDs from that point until all 50 are done** — inter-RD κ is
only meaningful if the two judgements are independent.

---

## 5. Export and analyse

Export **each project separately** from **Project → Export → JSON** (the full
format, not JSON-MIN). Or:

```bash
curl -s "$LS_URL/api/projects/<RD1_PID>/export?exportType=JSON" \
  -H "Authorization: Token $LS_TOKEN" -o rd1_export.json
curl -s "$LS_URL/api/projects/<RD2_PID>/export?exportType=JSON" \
  -H "Authorization: Token $LS_TOKEN" -o rd2_export.json
```

Merge them, attributing each file to one rater:

```bash
python src/pipelines/18-rd_judge_agreement.py import \
  --export RD1=rd1_export.json RD2=rd2_export.json
```

**Single-project layout instead?** Pass one file and split by user:

```bash
python src/pipelines/18-rd_judge_agreement.py import \
  --export ls_export.json \
  --rater-map rd1@hospital.ac.id=RD1 rd2@hospital.ac.id=RD2
```

`--rater-map` accepts whatever `completed_by` resolves to — email, username, or
numeric user id. Run once without it to see the raw identifiers, then re-run
with the mapping.

Either way the command reports how many units got 1 vs 2 annotations; units
with a single rater still contribute to judge-vs-RD but not to inter-RD. It
also warns if the same (unit, rater) pair arrives twice, which is the symptom
of passing the same export file under two labels.

Then:

```bash
python src/pipelines/18-rd_judge_agreement.py agreement
python src/pipelines/18-rd_judge_agreement.py report   # LaTeX table
```

`agreement` writes `data/18-rd_judge_agreement/agreement.json` and prints
inter-RD κ, judge-vs-RD κ, leniency, and the prevalence check.
`report` writes `agreement_table.tex`, ready to `\input` into the appendix.

### Smoke-test before the RDs start

```bash
python src/pipelines/18-rd_judge_agreement.py agreement --simulate
```

Fabricates two RDs by perturbing the judge scores so you can verify the whole
chain. Output goes to `agreement_simulated.json` and every table is stamped as
simulated. **Not a result** — do not publish it.

---

## 6. Reading the output

**Statistics reported per dimension per stratum:**

- `kappa_quadratic` — the headline. Quadratic weights are right for an ordinal
  Likert scale: scoring 4-vs-5 is a near miss, 1-vs-5 is a real disagreement.
- `kappa_unweighted`, `kappa_linear` — for comparability with papers that
  report plain κ (including the existing 0.652 judge-vs-judge figure, which is
  unweighted).
- `kappa_binary_acceptable` — κ after collapsing to *clinically acceptable*
  (≥ 4) vs not. The most interpretable number for a clinical audience:
  "do the judge and the dietitian agree on whether this answer could be shown
  to a clinician?"
- `krippendorff_alpha_ordinal` — handles ≥ 2 raters and missing cells, so it
  survives a third RD or an incomplete queue.
- `percent_agreement`, `percent_agreement_within_1`, `pabak` — see below.
- `mean_difference` — judge minus human. Positive = judge is more generous.

**Category boundaries** are pinned to the full 1–5 scale even when a rater never
uses some level, so the expected-agreement term does not drift between strata
and the numbers stay comparable across the table.

**All CIs** are percentile bootstrap over annotation units (2000 resamples),
resampling units so paired rater scores move together.

### The safety-dimension caveat — read before interpreting

Over the drawn sample, GPT-4o-mini scores **safety = 5 on 94% of units**. When
one category holds that much of the mass, Cohen's κ collapses toward 0 even
under near-perfect agreement, because expected agreement is already near 1.
This is the classic prevalence problem, not evidence of a bad judge.

The `sample` command warns when any dimension exceeds 80% single-category mass,
and `agreement` therefore always reports **percent agreement** and **PABAK**
next to κ for every comparison. For the safety dimension, lead with percent
agreement and PABAK and report κ alongside with the prevalence noted —
don't report a near-zero safety κ without that explanation.

### Expect wide intervals

With 50 units split three ways, per-stratum κ rests on ~16–17 units and the
bootstrap CI will be roughly ±0.2–0.3 wide. That is enough to support a claim
like "judge-vs-RD agreement is substantially lower on recommendation questions
than on factual ones" only if the gap is large. If the pooled result is
promising but the strata are inconclusive, extend to 100 units
(`--n 100`, same seed keeps the first 50 stable only if the pool is unchanged —
re-run and re-import the full draw rather than assuming a prefix).

---

## 7. Files

| Path | What |
|---|---|
| `src/pipelines/18-rd_judge_agreement.py` | Sampling, import, κ statistics, LaTeX |
| `docs/rd_annotation/rubric_id.md` | RD-facing scoring sheet, full (Indonesian) |
| `docs/rd_annotation/rubric_short.html` | One-screen version for the Label Studio instructions panel |
| `docs/rd_annotation/labeling_config.xml` | Label Studio interface |
| `docs/rd_annotation/docker-compose.yml` | Server deployment |
| `data/18-rd_judge_agreement/` | Sample, tasks, annotations, results |
