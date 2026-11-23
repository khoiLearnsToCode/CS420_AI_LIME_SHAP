## Slide 1

Good morning, everyone. We are Hoàng, Khôi and Việt, and our topic is LIME and SHAP, two popular methods for explaining the predictions of a machine learning model.

Our question is simple: if a model is accurate, can we trust its explanation? To find out, we tested LIME and SHAP on a breast-cancer classifier and asked three things. Do the two methods agree? Do they give the same answer every time we run them? And do the features they pick actually matter to the model?

Here is a preview. This is one tumour that our Random Forest got wrong. It says malignant with probability zero point seven four five, but the tumour is benign. On the left are LIME's top five features, and on the right are SHAP's top five. Only two of them match: mean area and mean radius. So for the same wrong prediction, the two methods tell different stories. The rest of the talk explains why, and how often this happens.


## Slide 2

We judge an explanation on three things, taken from the survey by Mersha and colleagues.

First, it should be useful: it shows which features push the prediction up or down.

Second, it should be stable: it stays the same when we only change the random seed or the background data.

Third, it should be faithful: if we remove the features it calls important, the prediction should change a lot.

Our data is the Breast Cancer Wisconsin dataset: five hundred sixty-nine tumours and thirty features. We trained two models: Logistic Regression, with ninety-eight point two percent test accuracy, and Random Forest, with ninety-five point six percent. Both explainers explain the probability of malignant. LIME uses five thousand samples, and KernelSHAP uses one hundred background rows.

We picked three test tumours. One is confidently correct: both models say malignant with probability one. One is borderline: Logistic Regression gets it wrong. And one is misclassified: Random Forest gets it wrong. That is the one from the title slide. We chose these because explanations matter most when the model is unsure or wrong.


## Slide 3

LIME explains one prediction at a time. It is post-hoc, so it runs after training. And it is model-agnostic, so it only calls the model and never looks inside it.

The idea is this: close to one input, even a complex model looks almost linear. So LIME does three things, shown here on the image of the car. First, it creates many slightly changed copies of the input. For an image, it greys out some regions. Second, it asks the model for a prediction on each copy, here the probability of "car". Third, it fits a small linear model to those predictions, and copies closer to the original count more.

That is the formula at the top. We look for the simple model g that matches f best near x. Pi is the weight, and it gets smaller with distance. The important setting is sigma, the kernel width. It decides what "near" means. The coefficients of g are the explanation.


## Slide 4

SHAP comes from game theory. Think of the features as players in a team, and the prediction as the prize. The Shapley value, phi, splits the prize fairly. For each feature, it averages how much that feature adds across all possible groups of the other features. That is the long sum in the formula.

The key result is Theorem 1 of Lundberg and Lee. It says Shapley values are the only explanation of this additive form with three properties. Local accuracy: the parts add up exactly to the prediction. Missingness: a feature that is missing gets zero. Consistency: if the model relies more on a feature, its value never goes down.

Here is how to read the figure. Start from phi zero, the average prediction on the background data. Each feature then pushes the prediction up or down, and the arrows end exactly at the model's output.


## Slide 5

These two methods are closely connected. LIME's weak point is that its kernel and sigma are picked by heuristics, so its weights don't have to add up to the prediction. Also, the LIME library splits each feature into value ranges, so a LIME weight refers to a range, not to a single value.

KernelSHAP uses the same recipe as LIME, but it chooses the kernel and the loss so that the result is exactly the Shapley value. In this figure from the SHAP paper, LIME, in green, settles on a value that is different from the true Shapley value, in black. KernelSHAP, in blue, reaches it.

But KernelSHAP has its own costs. It assumes the features are independent, it depends on the background data, and it is slower.


## Slide 6

Here is a quick check by hand. For a linear model, the SHAP value of a feature is simply its weight times how far the input is from the mean. Here the baseline, which is the prediction at the mean, is three, and the prediction for our input is nine. The three SHAP values are four, one and one. And three plus four plus one plus one is exactly nine, just as local accuracy promises.

One detail: feature two has a negative weight but a positive SHAP value. That is because its value is below the mean. So the sign of a SHAP value depends on both the weight and the input.


## Slide 7

Now LIME by hand, on a small non-linear function, with only six sample points. For each point, we compute the model's output, its distance to the input, and its weight. Then we fit weighted least squares.

LIME finds zero point three nine five for the first slope, almost the true value of zero point three nine three. For the second slope, it finds minus zero point one four nine, while the true value is minus zero point one nine seven. The sign is right, but it is off by about zero point zero five. Changing sigma moves the answer too. The lesson: with few points, LIME's answer depends on which points it samples. We will see this again in our experiments, as instability.


## Slide 8

First question: do the two methods agree? This table compares them for every model and test case. Overlap at five is the share of top-five features they have in common. Spearman rho compares their full rankings.

On the easy case, the overlap is zero point eight to one. On the misclassified case, it drops to zero point four: only two shared features out of five. Even simple Logistic Regression has a rank agreement of only about zero point two five on the hard cases. The good news is that when both methods pick the same feature, they always agree on its direction.

The last column also shows that on Random Forest, KernelSHAP is six to seven times slower than LIME.


## Slide 9

Let's zoom in on the hardest case, the tumour from the title slide. Random Forest says malignant with zero point seven four five, but it is benign. LIME is on the left, and SHAP is on the right. Please compare the order and the direction of the bars, not their length, because the two methods use different scales.

LIME's top feature, worst concave points, points towards benign, and the size features point towards malignant. SHAP's top features are mostly size features, and they all point towards malignant. So a doctor would get two different explanations for the same mistake.

SHAP still keeps its promise. The baseline, zero point three two six, plus the contributions, zero point four one nine, gives zero point seven four five. That is exactly the model's output.


## Slide 10

Second question: are they stable? We ran LIME with twenty different seeds, and SHAP with five different background sets. Jaccard measures how much the top five overlaps between runs, and top-one is how often the number-one feature stays the same.

Neither method wins. LIME's Jaccard is between zero point seven two and zero point nine four. SHAP is perfectly stable in some cases, but it drops to zero point five three in one case. That means about half of the top five changes between runs. Even the number-one feature can change on the easy case. Random Forest with LIME keeps the same top feature only sixty percent of the time, and Logistic Regression with SHAP only forty percent. We also tried tuning the background size and the kernel width, but Jaccard never went above zero point seven three.


## Slide 11

Third question: do the chosen features really matter? We take the top features from each method and remove them one by one, by setting them to the training mean. Then we watch the predicted probability drop. If the explanation is faithful, the probability should fall quickly. We compare this with removing features in random order.

Both methods clearly beat random order. But the gap between LIME and SHAP is at most zero point zero four after ten removals. With one seed and only twenty test tumours, we can't say which method is more faithful.

One caveat about this test: replacing a feature with its mean can create unrealistic tumours, and correlated features still carry each other's information.


## Slide 12

An important warning: these methods show what the model looks at, not what causes cancer. Radius, perimeter and area grow together, so which one gets the credit depends on the model and the background. The perturbed samples can be tumours that don't exist. On the wrong prediction, the explanation describes the model's mistake, not the disease. And we never change a real tumour, only numbers in the input.

Here is the classic example from the LIME paper. A model calls a husky a wolf because of the snow in the background. The explanation is correct about the model, but snow is not what makes an animal a wolf.


## Slide 13

So, back to our three questions.

Do LIME and SHAP agree? Not on hard cases. Only two of the top five match on the misclassified tumour.

Do they give the same answer every run? Not always. Jaccard drops to zero point five three.

Do the features matter? Yes, more than random. But we can't say which method is better.

Our conclusion: an accurate model does not guarantee a trustworthy explanation. If you use LIME or SHAP, report your seed, your background data and your kernel width. Run it several times and check stability. Be extra careful on borderline and wrong predictions. And say "the model relies on X", not "X causes Y".


## Slide 14

These are our references, and our notebook and results are on GitHub. We used Claude to help explain concepts and draft code and slides, and we checked the claims against the papers and the numbers against our results.


## Slide 15

Thank you for listening. We are happy to take your questions.
