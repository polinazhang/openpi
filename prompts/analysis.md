## Context
Read prompts/completed.md and documentation files it referred to 

## Objective
Conduct statistical analysis on three set of datasets.

## Metrics

We aim to use different static metrics to analyze the domain gaps between model pretrained distribution and dataset distribution. The four metrics of interest are cosine similarity, gradient guidance vector, perturbance norm (vision), and action displacement vector norm.
Context: prompts/static-inference-context.md, prompts/perturbance-calculation-formalization.md (perturbance norm), prompts/new-metric-definition.md (gradient guidance vector), prompts/edr-cosine.md (cosine similarity), and directly the formulation details in prompts/completed for displacement vector.

I want you to perform result post processing and analysis.

## Dataset Static Inference Results

Since these results are retrieved at different times, there're slight naming and saving directory inconsistency that I will explain here. It's only about which directory the latent files are stored -- it's always the exact same latent files following the requirements and documentation in prompts/completed.md being stored. If the information about saving directories below contradict with other documentations, follow this file since those other files aren't aware of historical changes.

The results of our interest are
- cosine (condition-inference only), saved in cosine/ for franka and openarm, while in perturbance-all/ for ood
- gradient guidance vector (condition-inference only), saved in gradient-inference/ for franka and openarm, and not saved for ood (just leave the results for this section blank)
- vision perturbance norm (condition-inference only), saved in perturbance-noise-displacement for franka, perturbance-noise for openarm, and perturbance-all for ood
- action displacement vector norm, saved in perturbance-noise-displacement for franka, perturbance-all for ood, and not saved for openarm (also leave the results for this section blank)

Detailed listings are below

A.
/coc/testnvme/xzhang3205/static/franka_full
- ignore the franka_object_action_ood and franka_object_vision_ood subfolders here

franka_full/<dataset>/cosine
franka_full/<dataset>/gradient-inference
franka_full/<dataset>/perturbance-noise-displacement (vision perturbance + action displacement)

some datasets contain more than 10 episodes (62 episodes for example). However, in data processing you should only strictly use episode 0-9.

B.
/coc/testnvme/xzhang3205/static/openarm_full

openarm_full/<dataset>/cosine
openarm_full/<dataset>/gradient-inference
openarm_full/<dataset>/perturbance-noise (vision perturbance only, no displacement)

some datasets contain more than 10 episodes. However, in data processing you should only strictly use episode 0-9. Throw a warning if you ever discover there's less than 10 episodes


C.
/coc/testnvme/xzhang3205/static/ood_full

ood_full/<dataset>/perturbance-all (vision perturbance + action displacement + cosine)

Use all episodes. One dataset has one episode only. This is expected.



## Data Processing

If the stored latents aren't exppressive enough, draw from the meta information in the result folders if necessary.

**vision perturbance norm**
The result values are the gradient norm $S_v := \|g_v\|_2$ $g_v := \nabla_{h_type} L$ for the vision embedding, across diffusion steps 1-10

**cosine**
The result values are the cosine similarities from condition-inference. It should be across diffusion steps 1-10 as well, and per layer.

**gradient-inference** should be l2 norm of the vector

**action displacement vector norm** should be l2 norm, for all 10 diffusion steps differently.


## Plots and tables your code needs to generate

* All tables should be printed in txt files containing the title describing what it contains, and in latex code, starting from \begin{table} and ending at \end{table}. Tables should follow each other by one spare line. If the code is run another time, overwrite the old txt and replace it with a new one. They should be ready to copy paste into overleaf as they are, so make sure to use \_ instead of _ 


### Tables

For each set (franka, openarm, ood), the following tables should be present and printed to franka.txt, openarm.txt and ood.txt:

For the first diffusion step
- cosine (final layer average) +- std
- cosine (all layers average) +- std
- vision perturbance norm +- std
- gradient-inference +- std
- action displacement vector norm +- std

Averaged across all ten diffusion steps
- cosine (final layer average) +- std
- cosine (all layers average) +- std
- vision perturbance norm +- std
- gradient-inference +- std
- action displacement vector norm +- std

Name these columns cosine final; cosine all; vision norm; velocity norm; displacement norm. Precision should go to two decimal points.

### Plots

For all plots, corresponding std should always be a soft region surronding the lineplot.

**5 main plots**
x axis: diffusion step (x ticks should be int)
y axis: cosine final; cosine all; vision norm; velocity norm; displacement norm

**additional plots for cosine** 10 plots + one gif plot
for each diffusion step
x axis: layer id (x ticks should be int)
y axis: cosine averaged per each layer, for that specific diffusion step

GIF plot: play those 10 plots sequentially at an adjustable interval

## TO DO 
- Read all instructions and code. Read also the files that the instructions refer to. 
- Write data post-processing scripts to process all three sets of results at once. Hard code all paths. All tables and plots should be saved to static_results under respective subfolders (create three for three datasets)