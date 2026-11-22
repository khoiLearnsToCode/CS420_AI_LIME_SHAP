# S04 LIME and SHAP: presentation script (English, 15 minutes)

Script for `S04_LIME_SHAP_slides_v2.html` (43 slides).
About 1,560 spoken words: roughly 12 minutes of speaking at 130 words per minute, plus about 1 minute for changing slides. That leaves 1–2 minutes of buffer inside the 15-minute limit.

- Slide numbers match the numbers in the bottom-right corner of each slide.
- Many slides repeat the same table or chart and only change the box at the bottom. Say one sentence and move on; those slides should take 10–15 seconds.
- Text in *[brackets]* is a cue, not something to say.
- Checkpoints (where you should be by): end of Part 1 at 2:00, Part 2 at 6:15, Part 3 at 7:30, Part 4 at 11:00, end at about 13:30.

---

## Part 1: The question (slides 1–4, about 2 minutes)

### Slide 1: Title

Good morning, everyone. We are Hoàng, Khôi and Việt, and our topic is LIME and SHAP, two popular methods for explaining machine learning predictions.

Our question: if a model is accurate, can we trust its explanation? We tested LIME and SHAP on a breast-cancer classifier and asked three things. Do they agree? Do they give the same answer every run? And do the features they pick really matter?

As a preview, on the right: for one tumour our Random Forest got wrong, only 2 of the top 5 features match between the two methods.

### Slide 2: We test each explanation on three things

We judge an explanation on three things. **Useful**: it shows which features push the prediction up or down. **Stable**: it stays the same when we only change the seed or the background data. **Faithful**: removing its top features changes the prediction a lot.

These match the criteria in the survey by Mersha et al., shown on the right. The key point: high accuracy does not mean the explanation is good.

### Slide 3: Our data, models and settings

We used the Breast Cancer Wisconsin dataset: 569 tumours and 30 features. Each measurement appears as a mean, an error and a "worst" value, and "worst" just means the largest. Class 0 is malignant.

Logistic Regression reaches 98.2% accuracy and Random Forest 95.6%. Both explainers explain the probability of malignant. LIME uses 5000 samples, and KernelSHAP uses 100 background rows.

### Slide 4: We look at 3 cases

We picked three test tumours: one **confidently correct**, one **borderline** that Logistic Regression gets wrong, and one **misclassified** that Random Forest gets wrong. Explanations matter most when the model is unsure or wrong.

---

## Part 2: Background (slides 5–14, about 4 minutes)

### Slide 5: LIME and SHAP explain single predictions

LIME and SHAP explain **one prediction at a time**. They are **post-hoc**: they run after training. They are **model-agnostic**: they only call the model and never look inside it.

The figure from the LIME paper shows the idea: a model predicts flu, LIME shows which symptoms support it, and a doctor decides whether to trust it.

### Slide 6: Our reference papers

We built on three papers: Ribeiro et al. for LIME, Lundberg and Lee for SHAP, and the survey by Mersha et al. for our evaluation criteria.

### Slide 7: LIME — how it works

LIME's idea: near one input, a complex model looks almost linear. It has three steps. **Perturb**: create 5000 points near the input and get the model's predictions. **Weight**: closer points count more. **Fit**: fit a small linear model, and its coefficients are the explanation.

In the figure, the curved background is the black box, the bold red cross is our input, and the dashed line is LIME's model. It fits well near the cross, but not everywhere.

### Slide 8: The same steps on an image

Here are the same steps on an image. LIME greys out parts of the car, records the probability of "car" for each new image, and fits a weighted linear model. The parts it keeps are the explanation.

### Slide 9: LIME symbols

These are the symbols. The most important one is σ, the kernel width. If σ is too small, the result is noisy. If it is too large, the explanation is no longer local. The guitar example shows LIME keeping only the parts of the image that support each class.

### Slide 10: LIME weak points

LIME has weak points. The kernel and σ are chosen by heuristics, so the weights don't have to add up to the prediction. In this figure from the SHAP paper, LIME, in green, settles on a value different from the true Shapley value. Also, the `lime` library splits features into ranges, which makes it harder to compare with SHAP.

### Slide 11: SHAP

SHAP comes from game theory. Think of the features as players in a team and the prediction as the prize. The Shapley value splits the prize fairly: it averages how much each feature adds across all possible groups of the other features.

### Slide 12: Theorem 1

The key result: Shapley values are the **only** additive explanation with three properties. **Local accuracy**: the parts add up to the prediction. **Missingness**: a missing feature gets zero. **Consistency**: if the model relies more on a feature, its value doesn't go down.

On the right, people and SHAP agree on a simple model, but LIME, in green, does not.

### Slide 13: How to read SHAP values

To read SHAP values: start from the baseline, which is the average prediction. Each feature pushes up or down, and the arrows end exactly at the prediction.

### Slide 14: How SHAP connects to LIME

In fact, KernelSHAP is LIME with a special kernel. The figure shows that the Shapley kernel puts most weight on very small and very large groups of features, and LIME's kernel doesn't. But KernelSHAP has its own limits: it assumes the features are independent, and it depends on the background data.

---

## Part 3: Two worked examples (slides 15–19, about 1.25 minutes)

### Slide 15: Worked example 1

A quick check with a linear model. The baseline is 3, the prediction is 9, and the SHAP values are 4, 1 and 1. 3 plus 4 plus 1 plus 1 is exactly 9.

### Slide 16: Why is φ₂ positive?

Notice that feature 2 has a negative weight but a positive SHAP value. That's because its value is below the mean. So the sign depends on both the weight and the input.

### Slide 17: Worked example 2

Now LIME by hand, with only 6 points. For each point we compute the model's output, its distance and its weight, then fit weighted least squares.

### Slide 18: The slopes LIME finds

LIME finds 0.395 for the first slope, almost the true 0.393, and minus 0.149 for the second, against a true minus 0.197.

### Slide 19: What we learn from it

So the first slope is almost exact, and the second has the right sign but is off. With few points, the answer depends on the sample. We will see this again as instability.

---

## Part 4: Results (slides 20–35, about 3.5 minutes)

### Slide 20: LIME and SHAP agree less on harder cases

This table compares the two methods. **Overlap at 5** is the share of top-5 features they have in common. **Spearman ρ** compares their full rankings.

### Slide 21: Different features, same sign

Overlap is 0.8 to 1.0 on the easy case, but only 0.4 on the misclassified one. The good news: when they pick the same feature, they always agree on its direction.

### Slide 22: Low agreement even for LR

Even simple Logistic Regression has low rank agreement on hard cases: ρ is only about 0.25.

### Slide 23: KernelSHAP is slower

And on Random Forest, KernelSHAP is 6 to 7 times slower than LIME.

### Slide 24: On the misclassified tumour

Now the hardest case: a benign tumour that Random Forest calls malignant with 0.745. LIME is on the left and SHAP on the right. Compare the order and direction of the bars, not their length.

### Slide 25: Two different stories

They tell different stories. LIME's top feature points towards **benign**. SHAP's top features are all size features, and they all point towards **malignant**. A doctor would get two different explanations of the same mistake.

### Slide 26: SHAP values still add up

SHAP still adds up: 0.326 plus 0.419 equals 0.745, the model's output.

### Slide 27: On the easy case, they mostly agree

And on the easy tumour, both methods agree on four of the top five. So they disagree mainly where we need explanations most.

### Slide 28: Running the same explanation again

Next, stability. We ran LIME with 20 seeds and SHAP with 5 background sets. **Jaccard** is the overlap of the top 5 between runs, and **top-1** is how often the number-one feature stays the same.

### Slide 29: Neither method is always more stable

Neither method wins. SHAP is perfectly stable in some cases but drops to 0.53 in another, which means about half the top 5 changes between runs.

### Slide 30: The #1 feature can change even on easy cases

Even the top feature can change. On the easy case, LIME agrees on it only 60% of the time, and SHAP only 40%.

### Slide 31: Tuning didn't help

We tried tuning the background size and the kernel width, but Jaccard never went above 0.73.

### Slide 32: Example — Random Forest on the borderline case

One example: with seed 0, LIME ranks worst concavity first. With seeds 1 to 5, it ranks worst texture first. Same model, same tumour, different seed.

### Slide 33: Deletion test — how it works

Last question: do the chosen features really matter? We remove the top features one by one and watch the probability drop. We compare this with removing features in random order.

### Slide 34: Both methods beat random, but no winner

Both methods beat random clearly. On Random Forest, after 10 removals the probability falls to about 0.6, against 0.83 for random order. But the gap between LIME and SHAP is at most 0.04, so with our small test we can't pick a winner.

### Slide 35: One problem with this test

One caveat: replacing a feature with its mean can create unrealistic tumours, and correlated features still carry each other's information.

---

## Part 5: Discussion and conclusion (slides 36–43, about 2.5 minutes)

### Slide 36: Feature importance is not a cause

One warning: these methods show what the **model** looks at, not what causes cancer. Correlated features share the credit. The samples can be unrealistic. On the wrong prediction, they explain the model's mistake, not the disease. And we never change a real tumour.

The husky example makes this clear. The model calls a husky a wolf because of the snow. The explanation is right about the model, but snow is not what makes a wolf.

### Slide 37: Checks we suggest before trusting an explanation

So we suggest five checks: a sanity check on a known answer, the local fit, deletion against random, stability across runs, and robustness to small changes. We fully did only the stability check. Trust an explanation only if it passes more than one check.

### Slide 38: Back to our three questions

Back to our questions. Do they agree? **Not on hard cases**: only 2 of 5 match. Are they stable? **Not always**: Jaccard drops to 0.53. Do the features matter? **Yes, more than random**, but we can't say which method is better. So an accurate model does not guarantee a trustworthy explanation.

### Slide 39: When using LIME or SHAP

Our advice: report your seed, background data and kernel width. Run it several times. Run the checks, especially on wrong predictions. And say "the model relies on X", not "X causes Y".

### Slide 40: Limitations and future work

Our limits: one dataset, two models, three cases, and no confidence intervals. Next, we want to add an insertion test, TreeSHAP and more runs.

### Slide 41: References

Our references and code are listed here. We used Claude as a helper and checked everything against the papers and our results.

### Slide 42: Questions

Thank you. We are happy to take your questions.

### Slide 43: Thank you

*[Leave on screen during Q&A.]*
