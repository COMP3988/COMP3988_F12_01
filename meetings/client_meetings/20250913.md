13 September 12pm Week 6
- little introduction on how we do the training
- the results obtained are more important 
- no time limit on the demonstration can be 5/10/15 minutes
- we are all not medical experts so we don't recognise the organs, so we compare the metric value and look at the boundaries of some organs and to look at if there's something here that has similar size - don't care too much about the details just if they look similar then that's enough and focus on the metrics
- compare our metrics with the recent literatures 
- successfully training a model and doing the transformation then SSIM can be a bit higher for ours personally - mainly depends on 2 things the metric value and the appearance of the generative image - ours is 0.5 so seems like we didn't find the best configuration or maybe some problems in our training
- seems like our generative image looks more similar to our MRI instead of our desired ground truth CT - some problems could be is issues in training or the model itself is not appropriate we need to check ourself and check other models at the same time
- UNet is not bad theoretically - we don't necessarily need long range dependencies - check if our training has any problems first 