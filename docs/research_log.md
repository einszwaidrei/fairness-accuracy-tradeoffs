# Research Log

## Purpose

This log documents how my Fairness–Accuracy Trade-offs project developed, from the initial baselines to the final comparison of eight model-strategy configurations.

I reconstructed it from my earlier notebooks after reorganizing the repository. The final notebooks show what I implemented, but they don't fully capture the intermediate results or the reasoning behind some of the methodological choices. I wanted to preserve that context without keeping multiple outdated versions of the same experiments in the main repository.

This is a retrospective account, not a day-by-day laboratory diary.

## 1. Establishing the Baselines

I started with two classifiers, Logistic Regression and Random Forest, using the ACSIncome task from the 2018 American Community Survey for Illinois.

I used a stratified 60/20/20 train-validation-test split with random seed 42. Before testing mitigation strategies, I needed a consistent way to compare predictive performance and group fairness.

I tracked five metrics:

- Accuracy
- Balanced Accuracy
- Equal Opportunity Difference (EOD)
- Demographic Parity Difference (DPD)
- False Positive Rate Gap

I chose absolute EOD as the primary fairness metric. It measures the absolute difference in true positive rates between the recorded SEX groups. In this study, that means comparing how often the models correctly identify individuals whose income exceeds $50,000 across the two groups.

I kept DPD and the FPR gap as secondary metrics because I didn't want to evaluate fairness through a single number. Different metrics capture different types of disparities, and improving one does not necessarily improve the others.

The baseline experiments gave me a reference point for evaluating whether subsequent mitigation methods actually changed these outcomes.

## 2. Reweighing: An Unexpected Result

Reweighing was the first mitigation strategy I evaluated. It changes the importance of training examples by assigning sample weights based on the relationship between the sensitive attribute and the target label.

The early validation results were particularly interesting because Logistic Regression and Random Forest responded differently.

| Model | Metric | Baseline | Reweighing |
|---|---|---:|---:|
| Logistic Regression | Accuracy | 0.8020 | 0.7994 |
| Logistic Regression | EOD | 0.0215 | 0.0349 |
| Logistic Regression | DPD | 0.1405 | 0.0923 |
| Logistic Regression | FPR Gap | 0.0515 | 0.0075 |
| Random Forest | Accuracy | 0.7868 | 0.7839 |
| Random Forest | EOD | 0.0294 | 0.0081 |
| Random Forest | DPD | 0.1305 | 0.1027 |
| Random Forest | FPR Gap | 0.0413 | 0.0203 |

For Logistic Regression, reweighing reduced DPD and the FPR gap, but EOD increased from 0.0215 to 0.0349. Random Forest showed a different pattern: all three measured fairness gaps decreased.

This was one of the most useful intermediate findings. Reweighing could improve some aspects of group fairness while worsening another, even within the same model.

It also made me more cautious about treating a mitigation algorithm as inherently fairer. The outcome depends on which metric matters and which classifier is being evaluated.

These were preliminary validation-set results, not the final held-out test results. I retained them here because they help explain why the later evaluation considered multiple fairness metrics rather than reporting EOD alone.

## 3. Adding Threshold Optimization

After reweighing, I added Fairlearn's `ThresholdOptimizer` to compare a postprocessing approach with the training-data intervention used by reweighing.

The experiment used:

- Constraint: `true_positive_rate_parity`
- Objective: `balanced_accuracy_score`
- Threshold fitting on the validation split
- A separate held-out test split for final evaluation

The distinction between these approaches matters. Reweighing modifies the training process through sample weights, while threshold optimization changes how a trained model's scores are converted into predictions.

I also had to account for a practical limitation: group-specific threshold optimization uses the sensitive attribute at prediction time. That may not be feasible or appropriate in every deployment setting, particularly when access to sensitive information is restricted.

This made the comparison more interesting than simply asking which method produced the lowest EOD. A method can look attractive in an experiment while introducing additional requirements for a real application.

## 4. Extending the Comparison to Exponentiated Gradient

An earlier version of the final-evaluation notebook compared six configurations: Baseline, Reweighing, and Threshold Optimization for each of the two classifiers.

I later expanded the study by adding Exponentiated Gradient, an in-processing fairness mitigation approach.

The implementation used Fairlearn's `TruePositiveRateParity` constraint with `eps=0.02` and `max_iter=50`.

This brought the total to eight configurations and allowed me to compare three different families of mitigation methods:

- **Preprocessing:** Reweighing
- **Postprocessing:** Threshold Optimization
- **In-processing:** Exponentiated Gradient

I considered this a useful extension because the methods intervene at different stages of the modeling pipeline. Comparing them under the same experimental setup provides a broader view of the available trade-offs.

## 5. Consolidating the Final Evaluation

Once all four strategies were implemented for both classifiers, I consolidated the results into a common evaluation workflow.

The final comparison included accuracy, balanced accuracy, EOD, DPD, and false positive rate gaps on the held-out test set.

I also added bootstrap confidence intervals for accuracy and EOD, followed by an accuracy-versus-EOD Pareto analysis.

One result worth highlighting is Random Forest with Exponentiated Gradient. Its test EOD increased from 0.0180 for the baseline to 0.0279 after mitigation.

In other words, adding a fairness constraint did not automatically produce a better outcome on the primary fairness metric.

The Logistic Regression results showed a different pattern. Threshold Optimization reduced EOD from 0.0342 to 0.0117, while accuracy decreased from 0.8081 to 0.8034.

These comparisons illustrate why I wanted to examine both predictive performance and fairness rather than ranking methods using a single metric.

The bootstrap analysis also helped me distinguish observed differences from stronger statistical claims. A lower point estimate does not, by itself, establish that one configuration is reliably superior.

Similarly, the Pareto frontier identifies nondominated configurations among the evaluated results. It is a descriptive comparison, not proof that those configurations would remain optimal on other datasets.


## Lessons Learned

Looking back at the experiments, four lessons stand out.

**1. Fairness metrics are not interchangeable.**

The early reweighing results made this particularly clear. A method can reduce demographic parity differences while increasing equal opportunity differences. Reporting only one metric can hide important changes elsewhere.

**2. The underlying classifier matters.**

The same mitigation strategy did not produce identical effects for Logistic Regression and Random Forest. I would be cautious about generalizing the effectiveness of a mitigation method from a single model.

**3. Small differences need careful interpretation.**

Some configurations differed by only a few thousandths in accuracy or EOD. Bootstrap intervals help describe uncertainty, but they do not remove every limitation of comparing models on a single dataset and split.

**4. There is no universally best mitigation strategy.**

The appropriate choice depends on the fairness objective, predictive performance requirements, the consequences of different prediction errors, and deployment constraints.

The project also has clear boundaries. All experiments used one ACSIncome task, one state, one year, and a limited set of models and mitigation configurations. The results provide evidence about these particular experiments, not a general ranking of fairness algorithms.

My main takeaway is that fairness mitigation is not simply about making a metric smaller. It requires understanding what that metric represents, what changes elsewhere in the system, and whether the resulting trade-off makes sense for the intended application.