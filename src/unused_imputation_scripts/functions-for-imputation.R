library(tidyverse)
library(mice)
library(VIM)

# Gives a pattern (missingness) plot, and a table with the variables sorted by number of missings (in console)
plot_missing <- function(dataset, name) {
  png(paste0(name,"_missingness_plot.png"), width = 1000, height = 700)
  aggr_plot <- aggr(dataset, col=c('#4cc9f0','#d90429'), numbers=TRUE, sortVars=TRUE, labels=names(dataset),cex.axis = .9, oma = c(10,5,5,3), gap=3, ylab=c("Histogram of missing data","Pattern"))
  dev.off()
}


# With the dataset as input, returns a prediction matrix as a dataframe
make_prediction_matrix <- function(dataset) {
  ini <- mice(dataset,maxit=0,seed=121212,remove_collinear = FALSE)
  pred <- as.data.frame(ini$predictorMatrix)
  return(pred)
}


# Takes the dataset, number of iterations, number of imputations and the prediction matrix as input. Returns the imputed sets.
do_imputation <- function(dataset, iterations, no_imputations, pred_matrix) {
  ptm <- proc.time()
  data_imp <- mice(dataset, maxit=iterations, seed=121212, m=no_imputations, predictorMatrix = as.matrix(pred_matrix))
  print(proc.time() - ptm)
  return(data_imp)
}


# After imputation plots
after_imp_plots <- function(dataset, name) {
  png(paste0(name,"_density.png"), width = 2000, height = 1000)
  print(densityplot(dataset))
  dev.off()
  
  
  pdf(file = paste0(name,"_iterations.pdf"))
  print(plot(dataset))
  dev.off()
  
  
  png(paste0(name,"_stripplot.png"), width = 2000, height = 1000)
  print(stripplot(dataset, pch = c(21, 20), cex = c(1, 1.5)))
  dev.off()
  
  
  png(paste0(name,"_BW.png"), width = 3000, height = 1500)
  print(bwplot(dataset), layout = c(7, 3))
  dev.off()
}


# BOXPLOT
make_boxplot <- function(name, original_df, imputed_df, das28crp_named) {
  org <- original_df %>%
    ungroup() %>%
    drop_na({{das28crp_named}}) %>%
    select({{das28crp_named}}) %>%
    mutate(group = "original data")
  
  
  
  imp <- imputed_df %>%
    ungroup() %>%
    drop_na({{das28crp_named}}) %>%
    select({{das28crp_named}}) %>%
    mutate(group = "imputed data")

  
  all_bp <- rbind(
    org,
    imp
  )
  
  
  bp_plot <- ggplot(all_bp, aes(x=group, y=.data[[das28crp_named]], fill=group)) +
    geom_boxplot() +
    scale_fill_manual(values=c("#E69F00", "#56B4E9")) + 
    theme_minimal() +
    labs(title="Boxplot imputed vs original data") +
    theme(axis.text.x = element_text(size = 14),
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 18),
          plot.title = element_text(size=22))

  
  png(paste0(name,"_boxplot.png"), width = 800, height = 500)
  print(bp_plot)
  dev.off()
  
}










