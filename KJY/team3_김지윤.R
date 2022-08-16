##
#https://www.kaggle.com/competitions/titanic

train <- read.csv('./train.csv', header=T)

## 데이터 파악
str(train)

# 나이 컬럼 결측치에 평균 넣기
train$Age[is.na(train$Age)] = mean(train$Age, na.rm = TRUE)

## Embarked 컬럼의 빈값을 NA로 처리
train$Embarked == ''
train$Embarked[is.na(train$Embarked)] <- sample(na.omit(train$Embarked), 2)
train$Embarked

## 데이터 summary
summary(train)



## 나이컬럼 범주화
Ages <- cut(train$Age, breaks=c(0, 10,20,30,40,50,60,Inf), right=F, 
            labels=c('Under 10', '10~20', '20~30', '30~40', '40~50', '50~60', 'over 60'))

## 범주화한 나이 Ages컬럼으로 추가하기
train$Ages <- Ages
train$Ages <- factor(train$Ages)


## 성별 컬럼 수치화해서 추가하기
train$Sex_factor <- factor(train$Sex, levels=c('female', 'male'), labels = c(0, 1))
train$Sex_factor <- as.numeric(train$Sex_factor)

##카이검정_survived:Sex
survivors_Sex <- table(train$Sex, train$Survived)
survivors_Sex_prop <- prop.table(survivors_Sex, margin=2)
survivors_Sex_prop

survivors_sex_bar <- barplot(survivors_Sex_prop*100, ylim=c(0, 100), legend=c('female', 'male'))
survivors_sex_bar
##카이검정_survived:Pclass
survivors_Pclass <- table(train$Pclass, train$Survived)
survivors_Pclass_prop <- prop.table(survivors_Pclass, margin=2)
survivors_Pclass_prop

survivors_pclass_bar <- barplot(survivors_Pclass_prop*100, ylim=c(0, 100), legend=c('1st', '2nd', '3rd'))
survivors_pclass_bar
##카이검정_survived:Ages
survivors_Ages <- table(train$Ages, train$Survived)
survivors_Ages_prop <- prop.table(survivors_Ages, margin=2)
survivors_Ages_prop

survivors_Ages_bar <- barplot(survivors_Ages_prop*100, ylim=c(0, 100), legend=c('under10', '10~20', '20~30', '30~40', '40~50', '50~60', 'over 60'))
survivors_Ages_bar
##카이검정_survived:Embarked
survivors_Embarked <- table(train$Embarked, train$Survived)
survivors_Embarked_prop <- prop.table(survivors_Embarked, margin=2)
survivors_Embarked_prop

survivors_Embarked_bar <- barplot(survivors_Embarked_prop*100, ylim=c(0, 100), legend=c('C', 'Q', 'S'))
survivors_Embarked_bar
## 독립성검정
chisq.test(survivors_Sex)
chisq.test(survivors_Pclass)
chisq.test(survivors_Ages)
chisq.test(survivors_Embarked)

## 두 변수간 관련 강도 확인
library(vcd)
assocstats(survivors_Sex)
assocstats(survivors_Pclass)
assocstats(survivors_Ages)
assocstats(survivors_Embarked)

## 두 변수간 관계 모자이크 플롯
mosaic(survivors_Sex, shade=TRUE, legend=TRUE)
mosaic(survivors_Pclass, shade=TRUE, legend=TRUE)
mosaic(survivors_Ages, shade=TRUE, legend=TRUE)
mosaic(survivors_Embarked, shade=TRUE, legend=TRUE)


## logistic regression
non=c('PassengerId', 'Name', 'Embarked', 'Cabin', 'Ticket', 'Ages', 'Sex', 'SibSp')
train_2 <- train[, !(names(train)%in% non)]
str(train_2)
titanic_logit <- glm(Survived~., data=train_2, family=binomial(link='logit'))
summary(titanic_logit)


## 예측
titanic_logit_pred <- predict(titanic_logit, data=train_2, type='response')
titanic_logit_pred <- factor(titanic_logit_pred>0.5, levels=c(TRUE, FALSE), labels=c('not survived', 'survived'))
titanic_logit_pred
table(titanic_logit_pred)

## 예측과 결과비교_혼동행렬
table(train_2$Survived, titanic_logit_pred, dnn=c('Actual', 'Predicted'))
mean(train_2$Survived == titanic_logit_pred)


library(pROC)
#install.packages('pROC')
x <- roc(Survived~titanic_logit$fitted.values,data=train)
roc_plot <- plot.roc(x, col='tomato',print.auc =TRUE,max.auc.polygon = TRUE)

