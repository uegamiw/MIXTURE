# ------------------------------------------------
#  Downstream task 
#   author: Wataru Uegami, MD
# ------------------------------------------------

# In this script, we first predict UIP, a type of interstitial pneumonia with 
# a poor prognosis, and then analyze the histological risk factor for overall 
# survival using the findings extracted by the previous step.

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(pROC))
suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(survminer))

R.version

save_result <- TRUE


# ------------------------------------------------
#  DATA LOADING & PREPARATION
# ------------------------------------------------
# All required data have to be placed in data_dir. All data will be exported 
# under the directory of data_out_dir.
# We have a feature files from MIXTURE model (features_MIXTURE.csv) and Non-
# integrated modes (features_non-integrated....csv).

# - features_MIXTURE.csv contains the number of each pathological findings 
#    (2.5x, 5x, 20x) in one file.
# - For non-integrated model, we only use the findings from 5x magnification. 
#    Number of the findings is 4-80.

## Load data
data_dir <- "/path/to/my/dir/data_for_R"
data_out_dir <- "/path/to/my/dir/pics_R"

r_csv <- function(fname, ...){
    suppressMessages(read_csv(paste(data_dir, fname, sep='/'), ...))
}

# Load the data for diagnosis & outcome
diagnosis <- r_csv('patients.csv', locale=locale(encoding="CP932"))

# Load the features from MIXTURE
feats <- r_csv('features_MIXTURE.csv')

# Load the result of 'Non-integrated' model
non_integrated <- list(
    'feats_k4'  = r_csv('features_non-integrated_k4_k4_k4.csv'),
    'feats_k8'  = r_csv('features_non-integrated_k4_k8_k8.csv'),
    'feats_k10' = r_csv('features_non-integrated_k10_k10_k10.csv'),
    'feats_k20' = r_csv('features_non-integrated_k20_k20_k20.csv'),
    'feats_k30' = r_csv('features_non-integrated_k30_k30_k80.csv'),
    'feats_k50' = r_csv('features_non-integrated_k30_k50_k80.csv'),
    'feats_k80' = r_csv('features_non-integrated_k30_k80_k80.csv')
)

# pre-process and merge
diagnosis <- diagnosis %>%
    filter(!is.na(UIP) & UIP != 'uncertain')

table(diagnosis$UIP)

# bind prognosis data and feature
helper1 <- function(df){
    df_tmp <- left_join(df, diagnosis, by = 'case') %>%
        filter(!is.na(UIP))
    df_tmp$UIP <- as.integer(df_tmp$UIP == 'UIP')
    df_tmp
}


# Apply to the "MIXTURE model"
feats <- helper1(feats)

# Apply to the "Non-integrated model"
non_integrated <- lapply(non_integrated, helper1)

# count num of records
cases_to_use <- intersect(feats$case, non_integrated$feats_k4$case)
length(cases_to_use)


helper2 <- function(df){
    filter(df, case %in% cases_to_use) %>% arrange(case)
}

feats <- helper2(feats)
non_integrated <- lapply(non_integrated, helper2)

# random forest algorithms doesn't work properly when colnames start with digit. We put 'M_' befor the colnames
helper3 <- function(df){
    val_cols <- df %>% select(contains('5x')) 
    colnames(val_cols) <- paste('M', colnames(val_cols), sep = '_')
    df %>%
        select(-contains('5x')) %>%
        bind_cols(val_cols)
}

non_integrated <- lapply(non_integrated, helper3)

# ------------------------------------------------
#  SPLIT TRAIN/VALID DATA
# ------------------------------------------------

train_id <- createDataPartition(feats$UIP, p=.7, list=F, times=1)

# split 'MIXTURE' -based model
train <- feats[train_id, ]
val <- feats[-train_id, ]

# split non-integrated model
helper4 <- function(df){
    df[train_id,]
}

helper5 <- function(df){
     df[-train_id,]
}

non_integrated_train <- lapply(non_integrated, helper4)
non_integrated_val <- lapply(non_integrated, helper5)

# ------------------------------------------------
#  PATIENT BACKGROUND
# ------------------------------------------------

train$train <- 'train'
val$train <- 'valid'
all <- bind_rows(train, val)

library(tableone)
CreateTableOne(vars = c('Age', 'UIP', 'time', 'Gender', 'event'), 
               factorVars = c('Gender', 'WSI', 'event'), strata = 'train', data = all)

t1 <- list()
t1$train <- all %>% filter(train == "train")
t1$valid <- all %>% filter(train == "valid")

CreateTableOne(vars = c('Age', 'UIP', 'time', 'Gender', 'event'), 
               factorVars = c('Gender', 'WSI', 'event'), strata = "UIP", data = t1$valid)


# ------------------------------------------------
#  UIP PREDICTION BY MACHINE LEARNING MODELS
# ------------------------------------------------
library(e1071)

# Prepare dataset for each combination of MIXTURE findings ----------------------
select2 <- function(...){
    train %>% select(UIP, ...)
}

data <- list()
data$m2m5m20 <- select2(contains('x'))
data$m2m5    <- select2(contains('2x'), contains('5x'))
data$m5m20   <- select2(contains('5x'), contains('20x'))
data$m2m20   <- select2(contains('2x'), contains('20x'))
data$m2      <- select2(contains('2x'))
data$m5      <- select2(contains('5x'))
data$m20     <- select2(contains('20x'))

# define RF -------------------------------
my_rf <- function(d){
    randomForest(UIP ~ ., data = d)
}

# define SVM ------------------------------
my_svm <- function(d){
    svm(UIP ~ ., data = d)
}

# Build random forest and svm models (MIXTURE) ----------------------------
rf_models <- lapply(data, my_rf)
svm_models <- lapply(data, my_svm)

# Build random forest and svm models (Non-integrated) --------------------
# Random Forest:
helper6 <- function(df){
    train_tmp <- df %>% select(contains('5x'), UIP)
    randomForest(UIP ~ ., data = train_tmp)
}

suppressWarnings(non_integrated_RF <- lapply(non_integrated_train, helper6))

# SVM: 
helper7 <- function(df){
    train_tmp <- df %>% select(contains('5x'), UIP)
    svm(UIP ~ ., data = train_tmp)
}

suppressWarnings(non_integrated_SVM <- lapply(non_integrated_train, helper7))

calcROC <- function(val_predict, df){
    suppressMessages(lapply(val_predict, function(p){roc(UIP ~ p, data = df, ci = T)}))
}

## Predicion -------------------------------------
  # Random Forest  (MIXTURE)
val_predict_rf <- lapply(rf_models, function(d){predict(d, val)}) 

  # ROC
ROC_MIXTURE_rf <- calcROC(val_predict_rf, val)


  # SVM (MIXTURE)
val2 <- val %>% select(contains('x'))
val_predict_svm <- lapply(svm_models, function(d){predict(d, val2)})

  # ROC 
ROC_MIXTURE_svm <- calcROC(val_predict_svm, val)

  # Random Forest (Non-integrated) 
non_integrated_predict_rf <- map2(non_integrated_RF, non_integrated_val, function(m,d){predict(m, d)})

  # ROC
non_integrated_UIP <- lapply(non_integrated_val, function(col){col$UIP})
ROC_non_integrated_rf <- suppressMessages(map2(non_integrated_UIP, non_integrated_predict_rf, function(x,y){roc(x~y, ci=T)}))

   # SVM (Non-integrated) 
non_integrated_val2 <- lapply(non_integrated_val, function(d){select(d, contains('x'))})
non_integrated_predict_svm <- map2(non_integrated_SVM, non_integrated_val2, function(m,d){predict(m,d)})

  # ROC
ROC_non_integrated_svm <- suppressMessages(map2(non_integrated_UIP, non_integrated_predict_svm, function(x,y){roc(x~y, ci=T)}))


# Feature Importance of Random Forest ------------------------------
lapply(rf_models, importance)

# Export all ROC plot -----------------------------------------------
helper8 <- function(dir_){
    function(x,y){
        fname <- paste0(data_out_dir, dir_, y, '.svg')
        svg(fname, width = 5.5, height = 5.5)
        plot(x, main=y)
        dev.off()
    }
}
mixture_rf <- helper8("/ROC/MIXTURE/rf")
mixture_svm <-  helper8("/ROC/MIXTURE/svm")
ni_rf <-  helper8("/ROC/Non_Integrated/rf")
ni_svm <-  helper8("/ROC/Non_Integrated/svm")


if(save_result){
    map2(ROC_MIXTURE_rf, names(ROC_MIXTURE_rf), mixture_rf)
    map2(ROC_MIXTURE_svm, names(ROC_MIXTURE_svm), mixture_svm)
    map2(ROC_non_integrated_rf, names(ROC_non_integrated_rf), ni_rf)
    map2(ROC_non_integrated_svm, names(ROC_non_integrated_svm), ni_svm)
}

# Save ROC data -----------------------
# Raw data
list(
    'mixture_rf' = ROC_MIXTURE_rf,
    'mixture_svm' = ROC_MIXTURE_svm,
    'non_integrated' = ROC_non_integrated_rf,
    'non_integrated' = ROC_non_integrated_svm
) %>% write_rds(paste0(data_out_dir, '/ROCdata.rds'))

# AUC and 95% CI -----------------------
auc_table <- function(roc_list){
    result <- data.frame(lapply(roc_list, function(x) x$ci)) %>% tibble() %>% t()
    colnames(result) <- c('inf95', 'auc', 'sup95')
    result <- data.frame(result)
    result$data <- names(roc_list)
    result
}
                                
mysaveCSV <- function(df, fname){
    write_csv(df, paste0(data_out_dir, '/', fname, '.csv'))
}

mysaveCSV(auc_table(ROC_MIXTURE_rf), 'ROC_MIXTURE_rf')
mysaveCSV(auc_table(ROC_MIXTURE_svm), 'ROC_MIXTURE_svm')
mysaveCSV(auc_table(ROC_non_integrated_rf), 'ROC_ni_rf')
mysaveCSV(auc_table(ROC_non_integrated_svm), 'ROC_ni_svm')        

# ------------------------------------------------
#  CALIBRATION PLOT (FIG S4)
# ------------------------------------------------

val$case2 <- paste0('case', 1:nrow(val))
plot_calib <- function(v, title = ''){
    val$predict <- v
    val %>%
        arrange(predict) %>%
        mutate(case2 = paste(case2, path_diag, sep = ', ')) %>%
        mutate(case2 = factor(case2, levels = case2)) %>%
        mutate(UIP = factor(ifelse(UIP==1, 'UIP', 'not UIP'), levels = c('UIP', 'not UIP'))) %>%
        ggplot(aes(x=case2, y=predict, col=UIP)) + geom_point() + theme_bw() +
            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                xlab('Original diagnosis') + ylab('score') + ggtitle(title)
    }
save_svg_calib <- function(g, title){
    fname <- paste0(data_out_dir, '/calib_plot/', title, '.svg')
    svg(fname, width=12, height=8)
    #par(new=TRUE)
    print(g)
    dev.off()
}

title <- '5x'
g <- plot_calib(val_predict_rf$m5, title)
save_svg_calib(g, title)
g

title <- '2x-5x-20x'
g <- plot_calib(val_predict_rf$m2m5m20, title)
save_svg_calib(g, title)
g

title <- '2x-5x'
g <- plot_calib(val_predict_rf$m2m5, title)
save_svg_calib(g, title)
g

title <- '5x-20x'
g <- plot_calib(val_predict_rf$m5m20, title)
save_svg_calib(g, title)
g

title <- '2x 20x'
g <- plot_calib(val_predict_rf$m2m20, title)
save_svg_calib(g, title)
g

title <- '2x'
g <- plot_calib(val_predict_rf$m2, title)
save_svg_calib(g, title)
g

title <- '20x'
g <- plot_calib(val_predict_rf$m20, title)
save_svg_calib(g, title)
g

# ------------------------------------------------
#  SURVIVAL ANALYSIS BY UIP PREDICTION
# ------------------------------------------------

# We set the a priori threshold as 0.5 to predict UIP.  

val_predict_rf_cat <- lapply(val_predict_rf, function(x){ifelse(x>0.5, 'UIP', 'not-UIP')})
val_predict_svm_cat <- lapply(val_predict_svm, function(x){ifelse(x>0.5, 'UIP', 'not-UIP')})

val$time_month <- val$time/30.4375

my_survplot <- function(f){
    ggsurvplot(f,  conf.int = F, risk.table = F, pval = T,
               censor.shape = 124, palette = 'jco', censor.size = 2, xlab='Month')
}

save_svg_surv <- function(g, title){
    fname <- paste0(data_out_dir, '/survplot/', title, '.svg')
    svg(fname, width=5.5, height=5.5)
    #par(new=TRUE)
    print(g)
    dev.off()
}

# 5x model
val$pred <- val_predict_rf_cat$m5
s_fit <- survfit(Surv(time_month, event) ~ pred, data = val)

g <- my_survplot(s_fit)
save_svg_surv(g, '5x')
g

survdiff(Surv(time, event) ~ pred, data = val)

summary(s_fit, times = seq(1, 70, by = 5))

# 5x - 20x model
val$pred <- val_predict_rf_cat$m5m20
s_fit <- survfit(Surv(time_month, event) ~ pred, data = val)

g <- my_survplot(s_fit)
save_svg_surv(g, '5x-20x')
g

survdiff(Surv(time, event) ~ pred, data = val)

summary(s_fit, times = seq(1, 70, by = 5)) # risk table

# 2x model
val$pred <- val_predict_rf_cat$m2
s_fit <- survfit(Surv(time_month, event) ~ pred, data = val)

g <- my_survplot(s_fit)
save_svg_surv(g, '2x')
g


# Difference between seveeral ROC
boot_n = 5000

# 5x vs 2.5x-5x-20x
roc.test(ROC_MIXTURE_rf$m5, ROC_MIXTURE_rf$m2m5m20, boot.n = boot_n, method = 'bootstrap')

# 5x vs 2x
roc.test(ROC_MIXTURE_rf$m5, ROC_MIXTURE_rf$m2, boot.n = boot_n, method = 'bootstrap')

# 5x (MIXTURE) vs 5x (Non-integrated)
roc.test(ROC_MIXTURE_rf$m5, ROC_non_integrated_rf$feats_k8, boot.n = boot_n, method = 'bootstrap')

prognosis <- r_csv('patients.csv',locale=locale(encoding="CP932"))
prognosis <- prognosis %>%
    filter(!is.na(event)) %>%
    select(case, event, time, UIP)

feats <- r_csv('features_MIXTURE.csv')
df_11C <- r_csv('features_non-integrated_k30_k80_k80.csv')

df <- feats %>%
    left_join(prognosis, by = 'case') %>%
    filter(!is.na(event))

df_11C <- df_11C%>%
    left_join(prognosis, by='case') %>%
    filter(!is.na(event))

cases_to_use <- intersect(df$case, df_11C$case)

df <- filter(df, case %in% cases_to_use) %>% arrange(case) %>% select(-case)
df_11C <- filter(df_11C, case %in% cases_to_use) %>% arrange(case) %>% select(-case)

nrow(df)

# ------------------------------------------------
#  COX PROPOTIONAL HAZARD ANALYSIS
# ------------------------------------------------
# Normalize
df_scale <- df %>%
    select(contains('_')) %>%
    scale() %>%
    data.frame()

df_scale$time <- df$time
df_scale$event <- df$event

coxph <- coxph(Surv(time, event) ~ ., data = df_scale)

suppressPackageStartupMessages(library(car))
vif(coxph) %>% data.frame()

# Quantitative correlation of extracted findings (Fig S5) -----------------

library(corrplot)

M <- df_scale %>%
    select(-time, -event) %>%
    cor()
corrplot(M, type='upper', tl.col = 'black')

svg(paste0(data_out_dir, '/corplot.svg'), width=10, height = 10)
corrplot(M, type='upper', tl.col = 'black')
dev.off()

# Histological risk factor in selected findings --------------------------

df_scale2 <- df_scale %>%
    select(-Cellular_fibrotic_IP_5x, -Pale_5x) %>%
    select(-Acellular_fibrosis_2x, -Accelular_fibrosis_5x, -Lymphoid_follicle_5x, -Near_Normal_2x)

coxph <- coxph(Surv(time, event) ~ ., data = df_scale2)
summary(coxph)

vif(coxph) %>% data.frame


# Histological risk factor of the histologically confirmed UIP case -------------------
# Subgroup analysis consisting of cases diagnosed as UIP by pathologists. 
# In the Cox proportional hazards model, the amount of fibroblastic foci was 
# confirmed to be a risk factor.(Table. S6)

df_scale$UIP <- df$UIP

df_scale2 <- df_scale %>%
    filter(UIP == 'UIP') %>%
    select(Cellular_fibrosis_2x, CellularIP_NSIP_5x, Edge_5x, Dense_fibrosis_20x, Immature_fibrosis_20x,
          Elastosis_20x, Fat_20x, Lymphocytes_20x, Mucous_20x, Resp_epithelium_20x, time, event) 

coxph <- coxph(Surv(time, event) ~ ., data = df_scale2)
summary(coxph)

# Histological risk factor of the histologically confirmed not-UIP case ------------------
# Subgroup analysis consisting of cases diagnosed as not UIP by pathologists. 
# In the Cox proportional hazards model, the aggregated lymphocytes were 
# identified as a risk factor. (Table S7)

df_scale$UIP <- df$UIP

df_scale2 <- df_scale %>%
    filter(UIP != 'UIP') %>%
    select(Cellular_fibrosis_2x, CellularIP_NSIP_5x, Edge_5x, Dense_fibrosis_20x, Immature_fibrosis_20x,
          Elastosis_20x, Fat_20x, Lymphocytes_20x, Mucous_20x, Resp_epithelium_20x, time, event) 

coxph <- coxph(Surv(time, event) ~ ., data = df_scale2)
summary(coxph)
