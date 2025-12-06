# Context-measure: Contextualizing Metric for Camouflage

![Camouflage Degree](/img/camouflage_degree.png)

## Quick Eval

### Overall

Before running the snippet, you only need to install the following minimium dependencies.

```shell
conda create -n cmeasure python=3.10
conda activate cmeasure
pip install -r requirements.txt
```

Then you can use the command 'python quick_eval.py' to start. Next is the quick start code:

### Evaluation

Running the `evaluate()` function produces our context-measure (general and camouflaged) along with other saliency-era evaluation metrics.

### Visualization

Running the `visualize()` function will generate camouflage heatmap.  All outputs are saved to the `vis_cd/` directory.

