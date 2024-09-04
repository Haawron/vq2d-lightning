# Lightning Implementation of VQ2D Models

<center>
<img src="resources/overview.png" alt="Repo Overview" style="width:450px;" />
</center>

## Currently Working on

### Prototyping

- [x] Dataset
  - [x] Build DINOv2 and check if outputs are reasonable: shouldn't generate same values
- [x] Metrics
- [ ] Evaluation
- [x] Loss
- [x] Training

### Smoke test (The Initial Sanity Check)

- [ ] Check if evaluation from official weight generates the same score.
- [ ] Check if `VQLoC-from-scratch` is correctly reproduced.

### Sanity Check Automation

- [ ] Build an unittest that runs training for a few tens of steps and check if the results are the same as before.
- [ ] Register this test to a github action, letting it submit an sbatch script when pushed.

### Implementations

#### LLaVA-NeXT-Video (7B) + LoRA as a unified encoder

- [ ] Check in advance with a smaller model like `BERT-base`.
- [ ] Do it.
