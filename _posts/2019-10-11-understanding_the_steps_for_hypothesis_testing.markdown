---
layout: post
title:      "Understanding the Steps for Hypothesis Testing "
date:       2019-10-11 21:12:19 +0000
permalink:  understanding_the_steps_for_hypothesis_testing
---

## (Specifically for t-tests)

I have recently conducted my first hypothesis test from start to finish in python using the Northwind Company fictional database.  

Starting this project, I had all the tools at my disposal, but I was having trouble understanding the steps to take to produce a thorough test from start to finish.  Now that I've completed my testing, I have a much better understanding of the steps to take and have decided to publish my method to help others who have similar concerns.

*Please note, for this blog I will focus on Student's t-test and Welch's t-test only.

### Step 1: Formulate your null and alternative hypothesis

* H0 = Null Hypothesis
* H1 = Alternative Hypothesis

### Step 2:  Set your significance level alpha

I will set alpha = 0.05 for the purpose of this blog.  Please note, this is a common significance level to use.

### Step 3: Clean and Explore Your Data

I will be skipping an explanation of this step since I'm focusing more on hypothesis testing then data cleaning/exploring databases in this blog.

### Step 4: Check Your t-test Assumptions:

#### Check if your samples have been drawn from a normal distribution

I found the best way to go about this is to plot your data.  There are a couple solutions if your samples are  not normally distributed.  

Please see the example below: 
```
import seaborn as sns
sns.distplot(ctrl, label = 'Control', kde = True)
sns.distplot(exp, label = 'Experimental', kde=True)
plt.title('Check for Normality')
plt.legend()
```

(ctrl = control group & exp = experimental group)

<blockquote class="imgur-embed-pub" lang="en" data-id="a/jHjTGsC" data-context="false" ><a href="//imgur.com/a/jHjTGsC"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

Here you can see my samples are positively skewed with a large right tail.  There are a few methods I can take to make the sample data more normal.  Some examples are below:

1. Apply the Central Limit Theorem = under many conditions, independent random variables summed together will converge to a normal distribution as the number of variables increases

2. Transform the data: for example, use log transformation

3. Remove outliers / clean the data further

For my example, I used the following code to apply the Central Limit Theorem

```
# Sampling With Replacement
import numpy as np

def get_sample(data, n):
    sample = []
    while len(sample) != n:
        x = np.random.choice(data)
        sample.append(x)
    return sample

# Generating a Sample Mean
def get_sample_mean(sample):
    return sum(sample)/len(sample)

# Creating a Sample Distribution of Sample Means
def create_sample_distribution(data, dist_size=100, n=30):
    sample_dist = [] 
    while len(sample_dist) != dist_size:
        sample = get_sample(data, n)
        sample_mean = get_sample_mean(sample)
        sample_dist.append(sample_mean)
    return sample_dist
		
ctrl_sample = create_sample_distribution(ctrl)
exp_sample = create_sample_distribution(exp)

sns.distplot(ctrl_sample, label = 'Control', kde = True)
sns.distplot(exp_sample, label = 'Experimental', kde=True)
plt.title('Samples_100_30 - Check for Normality')
plt.legend()
```

<blockquote class="imgur-embed-pub" lang="en" data-id="a/LsXSNyN" data-context="false" ><a href="//imgur.com/a/LsXSNyN"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

As you can see, using a distribution size of 100 and sample size of 30 brings me closer to normality.

#### Check if your samples are random and independent

The Central Limit Theorem coding in my example above guarantees random/independent data since I used np.random.choice.

#### Check if your samples are numeric and continuous values

The data in my example uses numeric and continuous values.

#### Check if your sample variances are equal or not

This is important when deciding which hypothesis test to use.  The Student's t-test assumes variances are equal.  If your variances are not equal, then you can use Welch's t-test.  

A simple way to check if your variances are equal is using the Levene Test

The Levene Test will test that the population variances are equal. If the resulting p-value < 0.05, the obtained differences in sample variances are unlikely to have occurred based on random sampling from a population with equal variances.

```
scipy.stats.levene(exp_sample,ctrl_sample)
```

<blockquote class="imgur-embed-pub" lang="en" data-id="a/d1dRxGa" data-context="false" ><a href="//imgur.com/a/d1dRxGa"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

Based on the samples, the p-value is > 0.05 and the null hypothesis of equal variances can be rejected.  Therefore,  it can be concluded that there is a difference between the variances in the population and you would choose to use Welch's t-test.

### Step 5: Perform your t-test

Welch's t-test: 
```
scipy.stats.ttest_ind(exp_sample,ctrl_sample, equal_var = False)
```

*If you use the Student's t-test then simply remove the equal_var line in the code above

If the p-value is < 0.05 then you can reject your null hypothesis.  If the p-value is > 0.05 then you cannot reject your null hypothesis.

### Step 6: Calculate Effect Size and Power

#### Effect Size:

Next you want to measure the effect size to determine how large of a statistically significant difference exists if you are able to reject your null hypothesis.  

A simple measurement for effect size is Cohen's d.  

```
import numpy as np
import scipy.stats

def Cohen_d(group1, group2):

    # Compute Cohen's d.

    # group1: Series or NumPy array
    # group2: Series or NumPy array

    # returns a floating point number 

    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    
    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)
    
    return d

group1 = np.array(exp_sample)
group2 = np.array(ctrl_sample)
effect_size= Cohen_d(group1,group2)
effect_size
```

For Cohen's d, a general rule of thumb for the effect size is:

* Small effect = 0.2
* Medium Effect = 0.5
* Large Effect = 0.8

#### Power:

Lastly, you'll want to look at your power to determine the chance of having a type 2 error.  This error is the probability that you fail to reject the null hypothesis when it is actually false.  Your power will be between the values of 0-1.  The closer you are to 1 means that the chance of having a type 2 error is very low.

To calculate power, you need to input your effect_size, alpha and sample size:

```
from statsmodels.stats.power import TTestIndPower, TTestPower
power_analysis = TTestIndPower()
power_analysis.solve_power(effect_size=effect_size, alpha=.05, nobs1=len(exp_sample))
```

### Step 7: Draw Conclusions

The last step is to draw conclusions and recommendations based on your findings!
