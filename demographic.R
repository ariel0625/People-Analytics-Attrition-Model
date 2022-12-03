install.packages("fmsb")
install.packages("corrplot")
library(corrplot)
library(ggplot2)
library(dplyr)
library(fmsb)



# load the dataset ＃descriptive用tableau即可
# regression with Years.of.Service & different metrics 
Demogra = read.csv("/Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589/Detail Active Report with Demographics 10.2022 .csv")

# show the first 6 rows of the dataset
head(Demogra)

# show the size of the dataset
dim(Demogra)
summary(Demogra)

#Age
Demogra<-mutate(Demogra, Age_range = ifelse(Demogra$Age %in% 70:60, "70-60", ifelse(Demogra$Age%in% 60:50, "60-50",
                                                                                    ifelse(Demogra$Age %in% 50:40, "50-40",
                                                                                           ifelse(Demogra$Age %in% 40:30, "40-30",ifelse(Demogra$Age %in% 30:20, "30-20", "70up"))))))                                                                                        
Demogra



plot(Demogra$Years.of.Service ~ Demogra$Age,
     main="Year of service for Simple Regression",
     xlab="Age", ylab="Year of service")


Demogra[sapply(Demogra, is.factor)] <- data.matrix(Demogra[sapply(Demogra, is.factor)])

model1<- lm(Demogra$Years.of.Service ~ Demogra$Age, data=Demogra)
summary(model1)


model2<- lm(Demogra$Years.of.Service ~ Demogra$Normal.Working.Hours, data=Demogra)
summary(model2)

model3<- lm(Demogra$Years.of.Service ~ Demogra$Job.Function.Name, data=Demogra)
summary(model3)

model4<- lm(Demogra$Years.of.Service ~ Demogra$Person.Gender, data=Demogra)
summary(model4)

model5<-lm(Demogra4$Years.of.Service~Demogra4$Person.Ethnicity, data=Demogra4)
summary(model5)



# load the dataset
# regression with manager promotiom & different metrics 
Demogra4 = read.csv("/Users/leechenhsin/Desktop/Study@USA/07_UW_School/IMT589/Detail Active Report with Demographics 10.2022 _new column.csv")
Demogra4[sapply(Demogra4, is.factor)] <- data.matrix(Demogra4[sapply(Demogra4, is.factor)])

#view updated data frame
Demogra4
head(Demogra4)


Demogra4$Manager=ifelse(Demogra4$Manager.Flag==3,1,0)
head(Demogra4)

#years of service
model1<-glm(Demogra4$Manager~Demogra4$Years.of.Service, data=Demogra4, family=binomial)
summary(model1)

#age
model2<-glm(Demogra4$Manager~Demogra4$Age, data=Demogra4, family=binomial)
summary(model2)

#Ethnicity-white
model3<-glm(Demogra4$Manager~Demogra4$Ethnicity, data=Demogra4, family=binomial)
summary(model3)

model3.1<-glm(Demogra4$Manager~Demogra4$Ethnicity==7, data=Demogra4, family=binomial)
summary(model3.1)

#Job function
model4<-glm(Demogra4$Manager~Demogra4$Job.Function.Name, data=Demogra4, family=binomial)
summary(model4)

#Person Gender(no_regression)
model5<-glm(Demogra4$Manager~Demogra4$Person.Gender, data=Demogra4, family=binomial)
summary(model5)



#multiple regression
head(Demogra4)
fit1<- lm(Demogra4$Manager ~ 1, data = Demogra4)
summary(fit1)


fit2 <- lm(Demogra4$Manager ~ Ethnicity, data = Demogra4)
summary(fit2)

fit3 <- lm(Demogra4$Manager ~ Ethnicity+ Person.Gender, data = Demogra4)
summary(fit3)

fit4 <- lm(Demogra4$Manager ~ Ethnicity+ Person.Gender+Age, data = Demogra4)
summary(fit4)


fit5 <- lm(Demogra4$Manager ~ Ethnicity+ Person.Gender+Age+Job.Function.Name, data = Demogra4)
summary(fit5)

fit6 <- lm(Demogra4$Manager ~ Ethnicity+ Person.Gender+Age+Job.Function.Name+Years.of.Service, data = Demogra4)
summary(fit6)

fit7 <- lm(Demogra4$Manager ~ Ethnicity+Age+Years.of.Service, data = Demogra4)
summary(fit7)

#Compare variables....
#fit definitions, impact factors, simple-->multiple p-value comparision---> final decision 

#pyhton output, fit7 prediction accuracy, confusion, impact....

anova(fit1,fit2,fit3,fit4,fit5,fit6, fit7)



#generation regression


v1<-Demogra %>% group_by(Age_range,Job.Function.Name) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Person.Ethnicity) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Person.Gender) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Generation) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Generation) %>% summarize(cont_service= sum(Years.of.Service) )

v2<-Demogra %>% group_by(Age_range,Normal.Working.Hours) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Business.Unit.Name) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Department.Name) %>% summarize(cont_service= sum(Years.of.Service) )

Demogra %>% group_by(Age_range,Department.Name) %>% summarize(cont_service= sum(Years.of.Service) )


mylogit <- glm(Years.of.Service ~Normal.Working.Hours, data = Demogra)
summary(mylogit)




























                                                                                                        