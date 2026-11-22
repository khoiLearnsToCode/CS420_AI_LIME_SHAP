# S04 LIME and SHAP: presentation script (English)

Script for `S04_LIME_SHAP_slides_v2.html` (43 slides).
About 3,600 spoken words, so roughly 27 minutes at a calm pace (130 words per minute).

- Slide numbers match the numbers in the bottom-right corner of each slide.
- Several slides repeat the same table or chart and only change the box at the bottom. On those slides, the script only covers the new box.
- Text in *[brackets]* is a cue, not something to say.
- To shorten the talk, the easiest cuts are the symbol slides (9, 11), the second LIME worked-example slide (18) and the runtime slide (23).

---

## Part 1: The question (slides 1–4)

### Slide 1: Title

Good morning, everyone. We are Hoàng, Khôi and Việt, and our topic is LIME and SHAP. These are two popular methods for explaining the predictions of machine learning models.

Our main question is on the screen: if a model is accurate, can we trust its explanation?

To answer it, we used LIME and SHAP on a breast-cancer classifier. We asked three questions. Do the two methods agree? Do they give the same answer every time we run them? And do the features they pick really matter to the model?

On the right is a preview. Our Random Forest got this tumour wrong. LIME and SHAP each give their five most important features, but only two features are on both lists. So the answer is not simple, and that is what this talk is about.

### Slide 2: We test each explanation on three things

We judge an explanation on three things.

First, it should be **useful**: it should show which features push the prediction up and which push it down.

Second, it should be **stable**: if we only change the random seed or the background data, the result should stay about the same.

Third, it should be **faithful**: if we remove the features it calls important, the prediction should change a lot.

These ideas come from the survey by Mersha et al. On the right are their computer-centred criteria. Our "stable" is their *consistency*, and our "faithful" is their *fidelity*.

The main point of this slide: high accuracy does not mean the explanation is good. Accuracy is about the prediction, not about the explanation.

### Slide 3: Our data, models and settings

We used the Breast Cancer Wisconsin dataset. It has 569 tumours and 30 features. The features are 10 measurements, like radius, texture and area, and each one appears 3 times: the mean, the standard error, and the "worst" value, which means the largest value. So when you hear "worst area" today, it simply means the largest area.

Class 0 is malignant. We used an 80/20 split, so 455 tumours for training and 114 for testing.

We trained two models. Logistic Regression reaches 98.2% accuracy, and Random Forest reaches 95.6%. So both models are very accurate.

Both explainers explain the same output: the probability of malignant. LIME uses 5000 samples and its default kernel width. KernelSHAP uses 100 background rows and 2000 coalitions.

### Slide 4: We look at 3 cases

We picked three test tumours with a fixed rule.

The first one is **confidently correct**: both models say malignant with probability 1, and they are right.

The second one is **borderline**: the tumour is benign, but Logistic Regression says 0.634 malignant, so LR gets it wrong.

The third one is the **misclassified** case: the tumour is benign, but Random Forest says 0.745 malignant, so RF gets it wrong.

We chose these cases because explanations are most important when the model is unsure or wrong.

---

## Part 2: Background (slides 5–14)

### Slide 5: LIME and SHAP explain single predictions of an already-trained model

Before the methods, here are four terms.

LIME and SHAP give **explanations**: they explain a decision, but the model itself stays a black box.

They are **local**: they explain one prediction at a time, not the whole model.

LIME and KernelSHAP are **model-agnostic**: they only call the model and never look inside it. Some SHAP versions, like Linear SHAP and Deep SHAP, are model-specific.

And they are **post-hoc**: they run after training, by changing the input and watching the output.

The figure below is from the LIME paper. A model predicts flu. LIME shows that sneeze and headache support the prediction, and "no fatigue" goes against it. With this, a doctor can decide whether to trust the model. This is the kind of explanation we are testing today.

### Slide 6: Our reference papers

We built the talk on three main papers.

Ribeiro et al., 2016, introduced LIME. Their goal was to explain any classifier so that users can decide whether to trust it. One weak point is that the kernel and its width are chosen by heuristics.

Lundberg and Lee, 2017, introduced SHAP. They showed that many attribution methods belong to one family, and that Shapley values are the only solution with certain good properties. One limit is that KernelSHAP assumes the features are independent and depends on the background data.

Mersha et al., 2024, is a survey of explainable AI. We took our evaluation criteria from this paper. We use them as a guide, not as exact formulas.

### Slide 7: LIME — how it works

Now LIME. The formula looks heavy, but the idea is simple: near one input, a complex model can be approximated by a straight line.

LIME has three steps.

One, **perturb**: create 5000 points near the input and ask the model for its prediction on each one.

Two, **weight**: points close to the input get a high weight, and points far away get a low weight.

Three, **fit**: fit a small linear model on these weighted points. Its coefficients are the explanation.

The figure from the LIME paper shows this. The pink and blue background is the black-box model's decision boundary, which is very curved. The bold red cross is the input we want to explain. The size of each point is its weight. The dashed line is LIME's linear model. It fits well near the cross, but not across the whole space. That is what "local" means.

### Slide 8: The same steps on an image

This figure, from Sadeghi et al., 2024, shows the same three steps on an image.

First, the car image is split into regions called superpixels. LIME greys out different groups of regions to create new images. For each new image, it records the model's probability of "car": 0.3, 0.85, 0.1, and so on. Then it fits a weighted linear model. The regions the model keeps form the explanation on the right.

In our project we use tabular data, not images, but the steps are exactly the same.

### Slide 9: LIME symbols

Here are the symbols. *[Point at each row briefly.]*

*f* is the black box. In our case, f of x is the probability of malignant, and x has 30 features.

*z* is a perturbed sample, and *z-prime* is its on/off version: each feature is either kept or removed.

*π* is the weight, and *σ* is the kernel width. If σ is too small, the result is noisy. If it is too large, the explanation is no longer local.

*G* is the family of linear models, and *Ω* keeps at most K features so that the explanation stays short.

The guitar figure shows what z-prime means for images. The model's top classes are electric guitar, acoustic guitar and Labrador. For each class, LIME keeps only the superpixels that support that class. For "electric guitar", it keeps the fretboard. For "Labrador", it keeps the dog's face.

### Slide 10: LIME weak points

LIME has three weak points.

First, the kernel and σ are chosen by heuristics, not by theory.

Second, because of this, the LIME weights do not have to add up to the prediction. The figure from the SHAP paper shows this. In this figure, green is LIME. Even with more samples, LIME settles on a value that is different from the true Shapley value, shown in black. KernelSHAP, in blue, reaches the true value.

Third, a point from our own experiment. The `lime` library splits each feature into ranges. So a LIME weight is about a range, like "worst area above 1000", not about the exact value. This makes LIME harder to compare with SHAP.

### Slide 11: SHAP

Now SHAP. SHAP comes from game theory. Think of the features as players in a team, and the prediction as the prize. The Shapley value is a fair way to split the prize.

For each feature *j*, we look at every possible group of the other features. We check how much the prediction changes when we add feature *j* to that group. The Shapley value is the weighted average of these changes.

*φ_j* is the contribution of feature *j*, in probability units, and *M* is the number of features, which is 30 here.

### Slide 12: Theorem 1

The main result of the SHAP paper is Theorem 1. Shapley values are the **only** additive explanation with three properties.

**Local accuracy**: the contributions add up exactly to the prediction.

**Missingness**: a feature that is missing gets zero.

**Consistency**: if a model relies more on a feature, that feature's SHAP value does not go down.

On the right, the SHAP paper tests this with people. The model is a simple sickness score that depends on fever and cough. People give both symptoms positive credit, and SHAP agrees with them. LIME, in green, gives them negative credit. So these properties matter in practice.

### Slide 13: How to read SHAP values

Reading SHAP values is easy.

*φ_0* is the baseline: the average prediction on the background data. A positive φ_j pushes the prediction above the baseline, and a negative one pushes it below.

And the sum, baseline plus all contributions, always equals the prediction.

The figure from the SHAP paper shows this. We start at the base value, E of f of z. Each feature adds an arrow. Blue arrows push the prediction up, and the red one pushes it down. The arrows end exactly at f of x.

### Slide 14: How SHAP connects to LIME

SHAP and LIME are closely related.

KernelSHAP is actually LIME with a specific kernel, loss and regulariser. With those choices, the result is exactly the Shapley values.

LIME picks these settings by heuristics, so it can break local accuracy.

But KernelSHAP has its own limits: it assumes the features are independent, and its result depends on the background data.

The figure compares the two kernels. The Shapley kernel, in blue, gives the most weight to very small and very large groups of features. LIME's kernel, in green, does not. This one difference explains a lot of the gap between the two methods.

---

## Part 3: Two worked examples (slides 15–19)

### Slide 15: Worked example 1 — SHAP values add up

Let's check this with numbers. We take a simple linear model: f of x equals 1 plus 2 x1 minus x2 plus 0.5 x3. The feature means are 1, 2 and 4, and our input is 3, 1 and 6.

For a linear model with independent features, SHAP has a closed form: the weight times the feature value minus its mean.

The baseline is f of the mean, which is 3. The prediction is 9. The contributions are 4, 1 and 1. And 3 plus 4 plus 1 plus 1 is 9, exactly the prediction. The waterfall chart on the right shows the same sum.

### Slide 16: Why is φ₂ positive?

One detail can be confusing. Feature 2 has a **negative** weight, minus 1, but its SHAP value is **positive**.

That is because x2 is 1, which is below its mean of 2. Minus 1 times minus 1 is plus 1. So the sign of a SHAP value depends on both the weight and the input value. A negative weight does not always mean a negative contribution.

### Slide 17: Worked example 2 — LIME by hand

Now LIME, by hand. The model is sigmoid of x1 squared minus x2, and we explain the point (1, 2), where the prediction is 0.2689.

To keep it small, we use 6 fixed sample points instead of 5000, a kernel width of 1, and no feature limit.

Take the first row of the table. The sample is (1.3, 2.2), and the model gives 0.3752. Its distance to our point is 0.361, so its weight is e to the minus 0.361 squared, which is 0.878. We do the same for all 6 points and then fit weighted least squares.

### Slide 18: The slopes LIME finds

Here is the result. For x1, LIME finds a slope of 0.395, and the true slope is 0.393, so it is almost exact. For x2, LIME finds minus 0.149, and the true slope is minus 0.197.

The last two columns show other kernel widths. The values change, but not by much in this example.

### Slide 19: What we learn from it

So LIME's first slope is almost exact. The second one has the right sign, but it is off by about 0.05.

The reason: 6 points cannot capture the curve of the model, so the answer depends on which points we sample. This is the same issue that we will see as instability in the real experiment.

---

## Part 4: Results (slides 20–35)

### Slide 20: LIME and SHAP agree less on harder cases

Now the real experiment. This table compares LIME and SHAP on our two models and three cases.

Two columns matter most. **Overlap at 5** is how many features the two top-5 lists share, divided by 5. The first number is from our main run, and the second is the average over repeated runs. **Spearman ρ** measures how similarly the two methods rank all 30 features.

### Slide 21: Different features, same sign

The pattern is clear. On the easy case, overlap is 0.8 to 1.0. On the misclassified case, it drops to 0.4, so only 2 of 5 features match.

There is also good news: when both methods pick the same feature, they always agree on its direction. They never disagree about whether a feature pushes towards malignant or benign.

### Slide 22: Low agreement even for LR

You might expect a simple linear model to be easy to explain. But Logistic Regression has the lowest rank agreement on the hard cases: ρ is only 0.24 and 0.26.

One possible reason is LIME's discretisation, which we mentioned before. But we did not test this, so it is only a guess.

### Slide 23: KernelSHAP is slower

On runtime, LIME is faster. On Random Forest, KernelSHAP takes 6 to 7 times longer, because it evaluates 2000 coalitions against 100 background rows. We timed this on one machine, so these numbers can vary by about 20%.

### Slide 24: On the misclassified tumour, LIME and SHAP give different explanations

Let's zoom in on the hardest case: the benign tumour that Random Forest calls malignant with probability 0.745.

On the left is LIME, and on the right is SHAP. The top-5 features are solid bars, and the features both methods picked are in bold. The two charts use different scales, so compare only the order and direction of the bars, not their length.

### Slide 25: Two different stories

The two methods tell different stories.

LIME's top feature, worst concave points, points towards **benign**, while the size features point towards malignant.

SHAP's top features are almost all size features: worst area, worst perimeter, worst radius. And all of them point towards **malignant**.

A doctor looking at these two charts would get two different explanations of the same mistake.

### Slide 26: SHAP values still add up

SHAP still keeps its promise. The baseline is 0.326, the contributions sum to 0.419, and 0.326 plus 0.419 is 0.745, exactly the model's output. That is local accuracy from Theorem 1, now on real data.

### Slide 27: On the easy case, they mostly agree

For contrast: on the confidently correct tumour, both methods put worst area, worst perimeter, worst radius and worst concave points at the top. Overlap is 0.8. So the disagreement appears mainly where we need explanations most: on hard and wrong predictions.

### Slide 28: Running the same explanation again can give a different top 5

Our second question is stability. We ran LIME with 20 different seeds, and SHAP with 5 different background sets.

**Jaccard** is the average overlap of the top-5 lists between pairs of runs, so 1 means identical lists. **Top-1** is the share of runs that agree on the number-one feature.

### Slide 29: Neither method is always more stable

Neither method wins. LIME's Jaccard goes from 0.72 to 0.94, and SHAP's goes from 0.53 to 1.00. SHAP is perfectly stable on Random Forest in two cases, but on the misclassified LR case it drops to 0.53. That means roughly half of the top-5 list changes between runs.

### Slide 30: The #1 feature can change even on easy cases

Even the most important feature can change. On the easy case, LIME on Random Forest agrees on the top feature only 60% of the time, and SHAP on Logistic Regression only 40% of the time. So if you run the explanation once and report the top feature, you may simply be reporting noise.

### Slide 31: Tuning didn't help

We tried to fix this by tuning. We changed the background size from 10 to 100 and the kernel width from 1 to 8. Jaccard never went above 0.73. So more tuning is not a simple fix.

### Slide 32: Example — Random Forest on the borderline case

Here is one example in detail. Each bar shows how often a feature is in the top 5: orange for LIME across 20 seeds, blue for SHAP across 5 backgrounds.

Look at the top feature. With seed 0, LIME ranks worst concavity first. With seeds 1 to 5, it ranks worst texture first. Same model, same tumour, only a different seed.

Another example: SHAP has mean concave points in its top 5 every time, but LIME only 10% of the time.

### Slide 33: Deletion test — how it works

Our third question: do the chosen features really matter? We used a deletion test.

We remove the top-ranked features one by one, by setting each one to its training mean, and we watch the probability of the predicted class. If the ranking is faithful, the probability should drop quickly. We compare this with removing features in random order, repeated 10 times. We used 20 test tumours.

### Slide 34: Both methods beat random, but no winner

Both methods do much better than random. On Random Forest, after 10 removals, the probability drops from 0.92 to about 0.6 with LIME or SHAP, but only to 0.83 with random order. Logistic Regression shows the same pattern.

But the gap between LIME and SHAP is tiny: at most 0.02 after 5 removals and 0.04 after 10. With one seed, 20 tumours and no confidence intervals, we cannot say which method is better.

### Slide 35: One problem with this test

This test also has a problem. Replacing a feature with its mean can create unrealistic tumours. And because features like radius, perimeter and area are strongly correlated, removing one still leaves its information in the others. So the test is useful, but it is not perfect.

---

## Part 5: Discussion and conclusion (slides 36–43)

### Slide 36: Feature importance is not a cause

One important warning: feature importance tells us what the **model** looks at. It does not tell us what causes cancer. Four reasons, all visible in our experiment, show why the two should not be confused.

**Correlated features**: radius, perimeter and area grow together, so which one gets the credit depends on the model and the background data.

**Unrealistic samples**: both methods change each feature on its own, which can create tumours that do not exist, like a large area with a small radius.

**Model mistakes**: on the misclassified tumour, the features explain why the *model* was wrong, not why a tumour is malignant.

**No real intervention**: we only change numbers in the model's input, not anything about a real tumour.

The husky example from the LIME paper shows this well. A model calls this husky a wolf, and LIME shows that it is looking at the snow. The explanation is correct about the model, but snow is obviously not what makes an animal a wolf.

### Slide 37: Checks we suggest before trusting an explanation

Based on all this, we suggest five checks before trusting an explanation.

One, a sanity check on a case where we know the answer, like our worked example.
Two, check the local fit: LIME's R-squared, and whether SHAP sums to the prediction.
Three, deletion and insertion tests against random order.
Four, top-k overlap across seeds and backgrounds.
Five, robustness to small input changes and to similar models.

The last column is honest about our own work: we fully did only the stability check, and some others only partly. Our rule: trust an explanation only if it passes more than one of these checks.

### Slide 38: Back to our three questions

So, back to our three questions.

Do LIME and SHAP agree? **Not on hard cases.** Only 2 of 5 top features match on the misclassified tumour.

Do they give the same answer every run? **Not always.** Mean Jaccard goes as low as 0.53.

Do the features they pick matter? **Yes, more than random.** Both beat random deletion, but we cannot say which one is better.

So, to answer the title: an accurate model does **not** automatically give a trustworthy explanation.

### Slide 39: When using LIME or SHAP

If you use LIME or SHAP, here is our practical advice.

Report the seed, the background data and the kernel width, because the result depends on them.

Run the explanation several times and report how stable it is.

Run the checks, especially for borderline and wrong predictions.

And choose your words carefully: say "the model relies on X", not "X causes Y".

### Slide 40: Limitations and future work

Our study has clear limits. We used one dataset, two models and three test cases. We have no confidence intervals, and there is no ground truth, so we cannot say which method is correct. Because LIME discretises features, we only compared ranks and signs. And we only used the probability scale and KernelSHAP.

Next, we would like to add an insertion test, LIME's R-squared, TreeSHAP and the log-odds scale, and more data and runs.

### Slide 41: References

These are our references. Our code, the CSV results and the figures are on GitHub at the link shown. We also used Claude to help explain concepts, write code and draft the report and slides. We checked every claim against the papers and every number against our CSV files.

### Slide 42: Questions

That's the end of our talk. We are happy to take your questions.

### Slide 43: Thank you

Thank you for your attention.
