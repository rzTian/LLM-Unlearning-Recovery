# About the repository

To get $\theta^{o}$ -- the model that memorizes the entire dataset -- run

```
sbatch train.sh
```

To perform unlearning, run

```
sbatch unlearn.sh

```

To evaluate the model, you can run either

```
sbatch eval.sh
```
for standard evaluation, or 

```
sbatch recover.sh
```
which implements specific decoding strategies to recover the unlearned knowledge.

In each shell script, you can replace the following by your own email address to track the queue process.

```
#SBATCH --mail-user=[Your email address]
```

# Tasks to be done (2025 April 10th)

- Partition the forget/retain set by attribute.
    - May consider randomly select several profiles (or just one or two profiles for an initial trial). Select one attribute (e.g., CVV) and unlearn that attribute for the selected profiles.
    - May consider selecting profiles that lie in different first-name category.

To do this, run `split_dataset.py` in the folder `data_generator`.


- Implement the following unlearning algorithms
    - [Gradient ascend with KL regularization](https://locuslab.github.io/tofu/)
    - [Preference Optimization](https://locuslab.github.io/tofu/)
    - [Negative preference optimization (NPO)](https://openreview.net/forum?id=MXLBXjQkmb#discussion) (To be done later...)

- Tune the hyper-parameters of these algorithms and make sure the unlearned model's performance on retain set does not drop too much.

- During unlearning, retain samples are usually drawn from the retain set. Collect the samples that are used in the unlearning process and evaluate the unlearned model on them.
    - Retain sample selection.
- Evaluate model's performance on profiles that share the same first names with the unlearned profiles.
- Collect instances on which model's performance drops most.
