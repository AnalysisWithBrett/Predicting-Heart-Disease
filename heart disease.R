# Heart disease analysis

# Libraries
library(ggplot2)
library(dplyr)
library(tidyverse)
library(class) #for KNN
library(gridExtra)
library(MASS)
library(adegenet)
library(ggalt)
library(Polychrome)
library(caret)
library(stringr)
library(ggforce)
library(psych)
library(nnet)
library(lmtest)
library(e1071) 
library(caTools) 
library(ClusterR) 
library(cluster) 
library(vegan)
library(readr)
library(ggord)# used for LDA
library(klaR)
library(zoo) #used for rolling averages
library(TTR) #used for exponential rolling averages
library(splines)
library(gam)
library(akima)
library(interp)
library ( leaps )
library(glmnet)
library(pls)
library(reshape2)
library(tree)
library(randomForest)
library(gbm)
library(writexl)
library(TTR)

#Data visualisation- creating function for your graph to look clean
theme.my.own <- function(){
  theme_bw()+
    theme(axis.text.x = element_text(size = 12, angle = 0, vjust = 1, hjust = 1),
          axis.text.y = element_text(size = 12, angle = 45),
          axis.title.x = element_text(size = 14, face = "plain"),             
          axis.title.y = element_text(size = 14, face = "plain"),             
          panel.grid.major.x = element_blank(),                                          
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank(),  
          plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), units = , "cm"),
          plot.title = element_text(size = 20, vjust = 1, hjust = 0.5),
          legend.text = element_text(size = 12, face = "italic"),          
          legend.title = element_text(size = 15, face = "bold.italic"),
          legend.background = element_rect(linetype = "solid", 
                                           colour = "black"))
}


# Data cleaning and manipulation

heart <- heart_2020

# Checking the data format

str(heart)
unique(heart$ChestScan)

# Converting some data into factors

factor_columns <- c("HeartDisease", "Smoking", "AlcoholDrinking", "Stroke",
                    "DiffWalking", "Sex", "AgeCategory", "Race",
                    "Diabetic", "PhysicalActivity", "GenHealth", "Asthma",
                    "KidneyDisease", "SkinCancer")

for (col in factor_columns) {
  heart[col] <- as.factor(heart[[col]])
}


# Check levels of factor variables
sapply(heart, function(x) length(unique(x[sapply(heart, is.factor)])))

# Checking the distribution of numeric variables

hist(log(heart$BMI))
hist((heart$SleepTime))
hist(heart$MentalHealth)
hist((heart$PhysicalHealth))

# Checking the distribution and correlation

pairs.panels(heart[,c(8,31,32,33,41,42)],
             gap = 0,
             bg = c("red", "blue")[heart$HadHeartAttack],
             pch = 21)

# Lasso selection
x <- model.matrix(HeartDisease ~ ., data = heart)[,-1 ] # Remove intercept column
y <- as.numeric(heart$HeartDisease)  # Assuming the response variable is categorical
dim(x)
length(y)
# Splitting data into training and testing dataset
set.seed(123)

# Define the size of the training and testing sets (1% of the total number of rows)
n <- nrow(heart)
train_size <- round(0.01 * n)
test_size <- round(0.01 * n)

# Randomly sample row indices for the training set
train_indices <- sample(1:n, train_size, replace = FALSE)

# Create the training set
train_set <- heart[train_indices, ]

# Remove the selected rows from the original dataset to obtain the remaining data
remaining_data <- heart[-train_indices, ]

# Randomly sample row indices for the testing set from the remaining data
test_indices <- sample(1:nrow(remaining_data), test_size, replace = FALSE)

# Create the testing set
test_set <- remaining_data[test_indices, ]

# Creating a grid of values ranging from λ = 10^10 to λ = 10^-2
grid <- 10^seq(10, -2, length = 100)

# Model selection with Lasso Model
lasso.mod <- glmnet(x[train_indices, ], y[train_indices], alpha = 1, lambda = grid)

#plotting the lasso model graph
plot ( lasso.mod )

plot(lasso.mod, xvar = "lambda", label = TRUE)


#Performing cross-validation
set.seed(123)
cv.out <- cv.glmnet (x[ train_indices , ], y[ train_indices ], alpha = 1)
plot(cv.out)

#choosing the best lambda
(bestlam <- cv.out $ lambda.min)


#testing the best lambda on the prediction
lasso.pred <- predict ( lasso.mod , s = bestlam ,
                        newx = x[ test_indices , ])
lasso.pred
#MSE for lasso
mean (( lasso.pred - y[test_indices] ) ^2)

#chooses the best model, which is model with 11 variables
out <- glmnet (x , y , alpha = 1, lambda = grid )
lasso.coef <- predict ( out , type = "coefficients",
                        s = bestlam ) [1:20 , ]
lasso.coef
#removing coefficients with 0 values
lasso.coef [ lasso.coef != 0]



####################################################################



#fitting random forest with p=6
#Note: regression is p/3 and classifaction is square root of p

rf.heart <- randomForest ( HeartDisease ~ Smoking + Stroke +
                           PhysicalHealth + DiffWalking + Sex +
                             AgeCategory, data = train_set , mtry = 4, importance = TRUE ,
                           n.trees = 1000)
rf.heart.predict <- predict ( rf.heart , newdata = test_set[,-1])


#Accuracy of test
mean(rf.heart.predict == test_set$HeartDisease)
table(predicted = rf.heart.predict, actual = test_set$HeartDisease)

#Accuracy
#Yes
46/(442+46)
#No
2904/(2904+9)

#We can view the importance of each variable
importance(rf.heart)

#Plotting to show the importance of variables
varImpPlot ( rf.heart )




###################################################



#KNN

# creating dataframe for KNN
heart1 <- heart

numeric_columns <- c("Smoking", "AlcoholDrinking", "Stroke",
                     "DiffWalking", "Sex", "AgeCategory", "Race",
                     "Diabetic", "PhysicalActivity", "GenHealth", "Asthma",
                     "KidneyDisease", "SkinCancer")


#Converting to numerical data
for (col in numeric_columns) {
  heart1[col] <- as.numeric(heart1[[col]])
}

training <- heart1[train_indices,]
testing <- heart1[test_indices,]

dim(testing)

# Finding the best k value for KNN
# Data preparation for k value
k_values <- c(1:100)

# Calculate accuracy for each k value

accuracy_values <- sapply(k_values, function(k) {
  knn.pred <- knn(train = training[,-1],  #you can also change this
                  test = testing[,-1], 
                  cl = training$HeartDisease, 
                  k = k)
  1 - mean(knn.pred != testing$HeartDisease)
})


# Create a data frame for plotting
accuracy_data <- data.frame(K = k_values, Full = accuracy_values)


# Plotting to find the best k
ggplot(accuracy_data, aes(x = K, y = Full)) +
  geom_line(color = "lightblue", size = 1) +
  geom_point(color = "lightgreen", size = 3) +
  labs(title = "Model Accuracy for Different K Values",
       x = "Number of Neighbors (K)",
       y = "Accuracy") +
  theme.my.own()


# Finding the best K
accuracy_data[which.max(accuracy_data$Full), ]

# Fitting to KNN model with k = k
(knn.pred <- knn(train = training[,c("Stroke", "Smoking", "PhysicalHealth",
                                     "DiffWalking", "Sex", "AgeCategory")], 
                 test = testing[,c("Stroke", "Smoking", "PhysicalHealth",
                                   "DiffWalking", "Sex", "AgeCategory")],
                 cl = training$HeartDisease, 
                 k = 1 ))


# Confusion Matrix 
#if there are zeroes around the diagonal then it has no error
table(predicted = knn.pred, actual = testing$HeartDisease) 


mean(testing$HeartDisease == knn.pred ) #measurement for error

#Accuracy
#Yes
29/(256+29)




######################################################


# Linear discriminant analysis
#Creating lda with the full model
lda_model <- lda(HeartDisease ~ Smoking + Stroke + PhysicalHealth +
                   DiffWalking + Sex + AgeCategory, data = training)
plot(lda_model)
summary(lda_model)
#Prediction of outcome
p <- predict(lda_model, testing)


# With testing data
lda.pred <- predict(lda_model, testing)$class
tab1 <- table(Predicted = lda.pred, Actual = testing$HeartDisease)
tab1

#Finding the lda accuracy
mean(testing$outcome == lda.pred )

#Accuracy for yes
58/(58+227)



#Quadratic discriminant analysis
qda.fit <- qda(HeartDisease ~ ., data = training)

qda.fit

#Making Predictions
qda.class <- predict ( qda.fit , testing[,-1])$class
table(predicted = qda.class ,actual = testing$HeartDisease)

mean(qda.class == testing$HeartDisease)

#Accuracy for yes
140/(140+145)









#####################################################

#Logistic regression
glm.fits <- glm(HeartDisease ~ Smoking + Stroke + PhysicalHealth +
                                   DiffWalking + Sex + AgeCategory, data = training ,
                family = binomial)

# Obtain predicted probabilities
glm.probs <- predict(glm.fits, testing, type = "response")

# Summary of the logistic regression model
summary(glm.fits)

# Making predictions
# Predict the class with the highest probability using the model object
predicted_classes <- ifelse(glm.probs > 0.5, "Yes", "No")

#Finding the overall accuracy
mean(predicted_classes == testing$HeartDisease)

#Checking the table
table(predicted = predicted_classes, actual = testing$HeartDisease)

#Accuracy for yes
27/(27+258)



