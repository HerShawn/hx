function WRA(precision,recall,fscore)
global totalGoodBbox totalPredBbox totalTrueBbox;
precision = [precision totalGoodBbox/totalPredBbox]
recall = [recall totalGoodBbox/totalTrueBbox]
if precision==0&&recall==0
    fscore=0
else
    fscore = [fscore 2*precision(end)*recall(end)/(precision(end)+recall(end))]
end
end