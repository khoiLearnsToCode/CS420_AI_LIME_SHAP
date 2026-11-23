# S04 LIME and SHAP: presentation script (English, 15 minutes)

Script for `S04_LIME_SHAP_slides_v3.html` (15 slides).
About 1,650 spoken words: roughly 12.5 minutes of speaking at 130 words per minute, plus a few seconds per slide change. That leaves about 2 minutes of buffer inside the 15-minute limit.

- Slide numbers count from the title slide (slide 1) to "Thank you" (slide 15).
- Text in *[brackets]* is a cue, not something to say.
- Checkpoints (where you should be by): end of Part 1 at 2:30, Part 2 at 6:00, Part 3 at 7:30, Part 4 at 11:30, end at about 13:00.
- If you are running late, shorten slide 7 (worked example 2) to its last two sentences.

---

## Part 1: The question (slides 1–2, about 2.5 minutes)

### Slide 1: Title

Good morning, everyone. We are Hoàng, Khôi and Việt, and our topic is LIME and SHAP, two popular methods for explaining the predictions of a machine learning model.

Our question is simple: if a model is accurate, can we trust its explanation? To find out, we tested LIME and SHAP on a breast-cancer classifier and asked three things. Do the two methods agree? Do they give the same answer every time we run them? And do the features they pick actually matter to the model?

Here is a preview. This is one tumour that our Random Forest got wrong: it says malignant with probability 0.745, but the tumour is benign. On the left are LIME's top 5 features, on the right SHAP's top 5. Only two of them match: mean area and mean radius. So for the same wrong prediction, the two methods tell different stories. The rest of the talk explains why, and how often this happens.

### Slide 2: Three checks, data and test cases

We judge an explanation on three things, taken from the survey by Mersha et al. 
1. **Useful**: it shows which features push the prediction up or down. 
2. **Stable**: it stays the same when we only change the random seed or the background data. 
3. **Faithful**: if we remove the features it calls important, the prediction should change a lot.

Our data is the Breast Cancer Wisconsin dataset: 569 tumours, 30 features. We trained two models: Logistic Regression, with 98.2%  test accuracy, and Random Forest, with 95.6%. Both explainers explain the probability of malignant. LIME uses 5000 samples, and KernelSHAP uses 100 background rows.

*[Point to the table.]* We picked three test tumours. One is **confidently correct**: both models say malignant with probability 1. One is **borderline**: Logistic Regression gets it wrong. And one is **misclassified**: Random Forest gets it wrong, the one from the title slide. We chose these because explanations matter most when the model is unsure or wrong.

---

## Part 2: How LIME and SHAP work (slides 3–5, about 3.5 minutes)

### Slide 3: LIME fits a simple linear model around one input

LIME explains **one prediction at a time**. It is **post-hoc**, so it runs after training, and **model-agnostic**, so it only calls the model and never looks inside it.

The idea: close to one input, even a complex model looks almost linear. So LIME does three things, shown on the image of the car. First, it creates many slightly changed copies of the input. For an image, it greys out some regions. Second, it asks the model for a prediction on each copy, here the probability of "car". Third, it fits a small linear model to those predictions, and copies closer to the original count more.

*[Point to the formula.]* That is the formula at the top: find the simple model g that matches f best near x, where π is the weight that gets smaller with distance. The important setting is σ, the kernel width. It decides what "near" means. The coefficients of g are the explanation.

### Slide 4: SHAP splits the prediction, and the parts add up

SHAP comes from game theory. Think of the features as players in a team and the prediction as the prize. The Shapley value, φ, splits the prize fairly: for each feature, it averages how much that feature adds across all possible groups of the other features. That is the long sum in the formula.

The key result, Theorem 1 of Lundberg and Lee, says Shapley values are the **only** explanation of this additive form with three properties. **Local accuracy**: the parts add up exactly to the prediction. **Missingness**: a feature that is missing gets zero. **Consistency**: if the model relies more on a feature, its value never goes down.

*[Point to the figure.]* To read it: start from φ₀, the average prediction on the background data. Each feature then pushes the prediction up or down, and the arrows end exactly at the model's output.

### Slide 5: KernelSHAP is LIME with a special kernel

These two methods are closely connected. LIME's weak point is that its kernel and σ are picked by heuristics, so its weights don't have to add up to the prediction. Also, the `lime` library splits each feature into value ranges, so a LIME weight refers to a range, not a single value.

KernelSHAP uses the same recipe as LIME, but chooses the kernel and the loss so that the result is exactly the Shapley value. *[Point to the figure.]* In this figure from the SHAP paper, LIME in green settles on a value different from the true Shapley value in black, while KernelSHAP in blue reaches it.

But KernelSHAP has its own costs. It assumes the features are independent, it depends on the background data, and it is slower.

---

## Part 3: Two worked examples (slides 6–7, about 1.5 minutes)

### Slide 6: Worked example 1 — SHAP values add up

A quick check by hand. For a linear model, the SHAP value of a feature is simply its weight times how far the input is from the mean. Here the baseline, the prediction at the mean, is 3, and the prediction for our input is 9. The three SHAP values are 4, 1 and 1. And 3 plus 4 plus 1 plus 1 is exactly 9, as local accuracy promises.

One detail: feature 2 has a **negative** weight but a **positive** SHAP value. That is because its value is below the mean. So the sign of a SHAP value depends on both the weight and the input.

### Slide 7: Worked example 2 — LIME by hand

Now LIME by hand, on a small non-linear function, with only 6 sample points. For each point we compute the model's output, its distance to the input and its weight, then fit weighted least squares. *[Point to the right table.]*

LIME finds 0.395 for the first slope, almost the true 0.393. For the second slope it finds minus 0.149, against the true minus 0.197: the right sign, but off by about 0.05. Changing σ moves the answer too. The lesson: with few points, LIME's answer depends on which points it samples. We will see this again in our experiments as instability.

---

## Part 4: Results (slides 8–11, about 4 minutes)

### Slide 8: LIME and SHAP agree less on harder cases

First question: do the two methods agree? This table compares them for every model and test case. **Overlap at 5** is the share of top-5 features they have in common. **Spearman ρ** compares their full rankings.

On the easy case, overlap is 0.8 to 1.0. On the misclassified case it drops to 0.4: only two shared features out of five. Even simple Logistic Regression has a rank agreement of only about 0.25 on the hard cases. The good news: when both methods pick the same feature, they always agree on its direction.

The last column also shows that on Random Forest, KernelSHAP is 6 to 7 times slower than LIME.

### Slide 9: On the misclassified tumour, they tell different stories

Let's zoom in on the hardest case, the tumour from the title slide. Random Forest says malignant with 0.745, but it is benign. LIME is on the left and SHAP on the right. Compare the order and direction of the bars, not their length, because the two methods use different scales.

LIME's top feature, worst concave points, points towards **benign**, and the size features point towards malignant. SHAP's top features are mostly size features, and they all point towards **malignant**. So a doctor would get two different explanations for the same mistake.

SHAP still keeps its promise: the baseline 0.326 plus the contributions, 0.419, gives 0.745, exactly the model's output.

### Slide 10: Running the same explanation again can give a different top 5

Second question: are they stable? We ran LIME with 20 different seeds, and SHAP with 5 different background sets. **Jaccard** measures how much the top 5 overlaps between runs, and **top-1** is how often the number-one feature stays the same.

Neither method wins. LIME's Jaccard is between 0.72 and 0.94. SHAP is perfectly stable in some cases, but drops to 0.53 in one, which means about half of the top 5 changes between runs. Even the number-one feature can change on the easy case: Random Forest with LIME keeps the same top feature only 60% of the time, and Logistic Regression with SHAP only 40%. We also tried tuning the background size and the kernel width, but Jaccard never went above 0.73.

### Slide 11: Deletion test — both beat random, but no winner

Third question: do the chosen features really matter? We take the top features from each method and remove them one by one, by setting them to the training mean, and watch the predicted probability drop. If the explanation is faithful, the probability should fall quickly. We compare this with removing features in random order.

*[Point to the curves.]* Both methods clearly beat random order. But the gap between LIME and SHAP is at most 0.04 after 10 removals. With one seed and only 20 test tumours, we can't say which method is more faithful.

One caveat about this test: replacing a feature with its mean can create unrealistic tumours, and correlated features still carry each other's information.

---

## Part 5: Discussion and conclusion (slides 12–15, about 1.5 minutes)

### Slide 12: Feature importance is not a cause

An important warning: these methods show what the **model** looks at, not what causes cancer. Radius, perimeter and area grow together, so which one gets the credit depends on the model and the background. The perturbed samples can be tumours that don't exist. On the wrong prediction, the explanation describes the model's mistake, not the disease. And we never change a real tumour, only numbers in the input.

*[Point to the husky.]* The classic example from the LIME paper: a model calls a husky a wolf because of the snow in the background. The explanation is correct about the model, but snow is not what makes an animal a wolf.

### Slide 13: Back to our three questions

So, back to our three questions. Do LIME and SHAP agree? **Not on hard cases**: only 2 of the top 5 match on the misclassified tumour. Do they give the same answer every run? **Not always**: Jaccard drops to 0.53. Do the features matter? **Yes, more than random**, but we can't say which method is better.

Our conclusion: an accurate model does not guarantee a trustworthy explanation. If you use LIME or SHAP, report your seed, background data and kernel width. Run it several times and check stability. Be extra careful on borderline and wrong predictions. And say "the model relies on X", not "X causes Y".

### Slide 14: References

These are our references, and our notebook and results are on GitHub. We used Claude to help explain concepts and draft code and slides, and we checked the claims against the papers and the numbers against our results.

### Slide 15: Thank you

Thank you for listening. We are happy to take your questions.

*[Leave on screen during Q&A.]*
