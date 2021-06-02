I’ve written about R-squared before and I’ve concluded that it’s not as intuitive as it seems at first glance. It can be a misleading statistic because [a high R-squared is not always good and a low R-squared is not always bad](https://blog.minitab.com/blog/adventures-in-statistics/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit). I’ve even said that [R-squared is overrated](https://blog.minitab.com/blog/adventures-in-statistics/how-high-should-r-squared-be-in-regression-analysis) and that [the standard error of the estimate (S)](https://blog.minitab.com/blog/adventures-in-statistics/regression-analysis-how-to-interpret-s-the-standard-error-of-the-regression) can be more useful.

Even though I haven’t always been enthusiastic about R-squared, that’s not to say it isn’t useful at all. For instance, if you perform a study and notice that similar studies generally obtain a notably higher or lower R-squared, you should investigate why yours is different because there might be a problem.

In this blog post, I look at five reasons why your R-squared can be too high. This isn’t a comprehensive list, but it covers some of the more common reasons.

## Is A High R-squared Value a Problem?

![Very high R-squared](https://blog.minitab.com/hubfs/Imported_Blog_Media/highr_sq.png)A very high R-squared value is not necessarily a problem. Some processes can have R-squared values that are in the high 90s. These are often physical process where you can obtain precise measurements and there's low process noise.

You'll have to use your subject area knowledge to determine whether a high R-squared is problematic. Are you modeling something that is inherently predictable? Or, not so much? If you're measuring a physical process, an R-squared of 0.9 might not be surprising. However, if you're predicting human behavior, that's way too high!

Compare your study to similar studies to determine whether your R-squared is in the right ballpark. If your R-squared is too high, consider the following possibilities. To determine whether any apply to your model specifically, you'll have to use your subject area knowledge, information about how you fit the model, and data specific details.

## Reason 1: R-squared is a biased estimate

The R-squared in your regression output is a biased estimate based on your sample—it tends to be too high. This bias is a reason why some practitioners don’t use R-squared at all but use adjusted R-squared instead.

R-squared is like a broken bathroom scale that tends to read too high. No one wants that! Researchers have long recognized that regression’s optimization process takes advantage of chance correlations in the sample data and inflates the R-squared.

Adjusted R-squared does what you’d do with that broken bathroom scale. If you knew the scale was consistently too high, you’d reduce it by an appropriate amount to produce a weight that is correct on average.

Adjusted R-squared does this by comparing the sample size to the number of terms in your regression model. Regression models that have many samples per term produce a better R-squared estimate and require less shrinkage. Conversely, models that have few samples per term require more shrinkage to correct the bias.

For more information, read my posts about [Adjusted R-squared](https://blog.minitab.com/blog/adventures-in-statistics/multiple-regession-analysis-use-adjusted-r-squared-and-predicted-r-squared-to-include-the-correct-number-of-variables) and [R-squared shrinkage](https://blog.minitab.com/blog/adventures-in-statistics/r-squared-shrinkage-and-power-and-sample-size-guidelines-for-regression-analysis).

## Reason 2: You might be overfitting your model

An overfit model is one that is too complicated for your data set. You’ve included too many terms in your model compared to the number of observations. When this happens, the regression model becomes tailored to fit the quirks and random noise in your specific sample rather than reflecting the overall population. If you drew another sample, it would have its own quirks, and your original overfit model would not likely fit the new data.

Adjusted R-squared doesn't always catch this, but [predicted R-squared](https://blog.minitab.com/blog/adventures-in-statistics/multiple-regession-analysis-use-adjusted-r-squared-and-predicted-r-squared-to-include-the-correct-number-of-variables) often does. Read my post about [the dangers of overfitting your model](https://blog.minitab.com/blog/adventures-in-statistics/the-danger-of-overfitting-regression-models).

## Reason 3: Data mining and chance correlations

If you fit many models, you will find variables that appear to be significant but they are correlated only by chance. While your final model might not be too complex for the number of observations (Reason 2), problems occur when you fit many different models to arrive at the final model. Data mining can produce [high R-squared values even with entirely random data](https://blog.minitab.com/blog/adventures-in-statistics/four-tips-on-how-to-perform-a-regression-analysis-that-avoids-common-problems)!

Before performing regression analysis, you should already have an idea of what the important variables are along with their relationships, coefficient signs, and effect magnitudes based on previous research. Unfortunately, recent trends have moved away from this approach thanks to large, readily available databases and automated procedures that build regression models.

For more information, read my post about using [too many phantom degrees of freedom](https://blog.minitab.com/blog/adventures-in-statistics/beware-of-phantom-degrees-of-freedom-that-haunt-your-regression-models).

## Reason 4: Trends in Panel (Time Series) Data

If you have time series data and your response variable and a predictor variable both have significant trends over time, this can produce very high R-squared values. You might try a [time series analysis](http://support.minitab.com/en-us/minitab/17/topic-library/modeling-statistics/time-series/basics/time-series-analyses-in-minitab/), or including time related variables in your regression model, such as [lagged](http://support.minitab.com/en-us/minitab/17/topic-library/minitab-environment/calculator-and-matrices/column-calculator-functions/lag-function/) and/or [differenced](http://support.minitab.com/en-us/minitab/17/topic-library/minitab-environment/calculator-and-matrices/column-calculator-functions/differences-function/) variables. Conveniently, these analyses and functions are all available in [Minitab statistical software](http://www.minitab.com/en-us/products/minitab/).

## Reason 5: Form of a Variable

It's possible that you're including different forms of the same variable for both the response variable and a predictor variable. For example, if the response variable is temperature in Celsius and you include a predictor variable of temperature in some other scale, you'd get an R-squared of nearly 100%! That's an obvious example, but you can have the same thing happening more subtlety.