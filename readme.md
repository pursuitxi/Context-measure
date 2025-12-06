<h1 align="center">Context-measure: Contextualizing Metric for Camouflage</h1>
<div align='center'>
    <a target='_blank'><strong>Chen-Yang Wang</strong></a><sup> 1</sup>,&thinsp;
    <a href= 'https://scholar.google.com/citations?user=oaxKYKUAAAAJ' target='_blank'><strong>Gepeng Ji</strong></a><sup> 1</sup>,&thinsp;
    <a target='_blank'><strong>Song Shao</strong></a><sup> 2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=huWpVyEAAAAJ' target='_blank'><strong>Ming-Ming Cheng</strong></a><sup> 1,2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=kakwJ5QAAAAJ' target='_blank'><strong>Deng-Ping Fan</strong></a><sup> 1,2*</sup>,&thinsp;
</div>

<div align='center'>
    <sup>1 </sup>VCIP, CS, Nankai University&ensp;  <sup>2 </sup>Chongqing Changan Wangjiang Industrial Group Co., Ltd.&ensp;
</div>

<div align="center" style="display: flex; justify-content: center; flex-wrap: wrap; margin-top: 50px;">
  <a href=https://github.com/pursuitxi/Context-measure''><img src='https://img.shields.io/badge/arXiv-Paper-red'></a>&ensp; 
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-yellow'></a>&ensp; 
</div>

![Camouflage Degree](/img/camouflage_degree.png)

## :rocket: Quick Eval

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

