import p4_utils
from scipy.stats import linregress

def plot_pred_non_redundancy(gold, pred, model='SVR'):
    line = linregress(gold, pred)
    label = "slope=%4.2f intercept=%4.2f rvalue=%4.2f pvalue=%4.2f stderr=%4.2f" % \
        (line.slope, line.intercept, line.rvalue, line.pvalue, line.stderr)
    slope = line.slope
    intercept = line.intercept
    heading = label
    subheading = ''
    x = "Test non-redundancies"
    y = "Predicted test non-redundancies"
    plotName = "tests/p4-1_"+model+"_test_non_redundancies_vs_predicted_test_non_redundancies"
    p4_utils.plot_compare(gold, pred, slope, intercept, heading, subheading, xlabel=x, ylabel=y, plotName=plotName)

def plot_pred_fluency(gold, pred, model='SVR'):
    line = linregress(gold, pred)
    label = "slope=%4.2f intercept=%4.2f rvalue=%4.2f pvalue=%4.2f stderr=%4.2f" % \
        (line.slope, line.intercept, line.rvalue, line.pvalue, line.stderr)
    slope = line.slope
    intercept = line.intercept
    heading = label
    subheading = ''
    x = "Test fluencies"
    y = "Predicted test fluencies"
    plotName = "tests/p4-2_"+model+"_test_fluencies_vs_predicted_test_fluencies"
    p4_utils.plot_compare(gold, pred, slope, intercept, heading, subheading, xlabel=x, ylabel=y, plotName=plotName)
