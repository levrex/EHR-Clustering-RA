createTableOne <- function(dat, dem, setmatch){
  setnames <- unique(setmatch$Set)
  
  out <- lapply(setnames, function(x){
    setdat <- dat[dat$ID %in% setmatch[setmatch$Set == x, "ID"],]
    demdat <- dem[dem$SUBJECT_ID %in% setmatch[setmatch$Set == x, "ID"],]
    
    res1 <- nrow(setdat)
    res2 <- length(ids)
    res3 <- length(unique(setdat$PheCode))
    
    res4 <- round(table(demdat$SEX)["C46110"]/nrow(demdat),2)
    res5 <- round(table(demdat$RACE)["C41261"]/nrow(demdat),2)
    res6 <- round(table(demdat$ETHNICITY)["C17459"]/nrow(demdat),2)
    
    tmp <- setdat[!duplicated(setdat$ID), "Age"]
  
    res7 <- round(median(tmp, na.rm = T))
    res8 <- round(range(tmp, na.rm = T)[1])
    res9 <- round(range(tmp, na.rm = T)[2])
  
    tmp <- split(setdat$PheCode, as.factor(setdat$ID))
    
    res10 <- median(sapply(tmp, length), na.rm = T)
    res11 <- range(sapply(tmp, length), na.rm = T)[1]
    res12 <- range(sapply(tmp, length), na.rm = T)[2]
    
    tmp <- lapply(tmp, unique)
    
    res13 <- median(sapply(tmp, length), na.rm = T)
    res14 <- range(sapply(tmp, length), na.rm = T)[1]
    res15 <- range(sapply(tmp, length), na.rm = T)[2]
    
    tmp <- split(setdat$Age, as.factor(setdat$ID))
    tmp <- sapply(tmp, function(y){
      ((max(y) - min(y))*365)+1
    })
    
    res16 <- round(median(tmp, na.rm = T))
    res17 <- round(range(tmp, na.rm = T)[1])
    res18 <- round(range(tmp, na.rm = T)[2])
    
    resfull <- c(res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13, res14, res15, res16, res17, res18)
    names(resfull) <- c("Entries", "IDs", "PheCodes", "PropFem", "PropWhite", "PropHisp", "PPAgeMed", "PPAgeMin", "PPAgeMax", "PPEntriesMed", "PPEntriesMin", "PPEntriesMax", "PPPheCodesMed", "PPPheCodesMin", "PPPheCodesMax", "PPFollowUpMed", "PPFollowUpMin", "PPFollowUpMax")
    
    return(resfull)
  })
  
  restable <- do.call("rbind", out)
  rownames(restable) <- unique(setmatch$Set)
  return(restable)
}

#requires reshape2, ggplot2 & viridis
createClusterSetProp <- function(dat, setmatch, flip=FALSE){
  csm <- merge(setmatch, dat[,c("ID", "Cluster")], by = "ID", all.y = F)
  csm <- csm[!duplicated(csm$ID),]
  csm$Set <- as.factor(csm$Set)
  if(!flip){
    tmp <- split(csm$Set, as.factor(csm$Cluster))
    res <- lapply(tmp, function(x){
      table(x)/length(x)
    })
    res <- do.call("rbind", res)
    res <- melt(res)
    colnames(res) <- c("Cluster", "Set", "Proportion")
    res$Cluster <- as.character(res$Cluster)
  
    p <- ggplot(res, aes(x = Cluster, y = Set, fill = Proportion)) +
    geom_tile() +
    theme_classic() +
    scale_fill_viridis(option = "magma") +
    theme(text = element_text(face = "bold",
                              size = 15,
                              colour = "black"),
          line = element_line(colour = "black"),
          axis.text = element_text(colour = "black"),
          axis.text.x = element_blank()) +
    labs(title = "Contribution of datasets to each cluster",
         x = "",
         y = "")
  } else {
    tmp <- split(csm$Cluster, as.factor(csm$Set))
    res <- lapply(tmp, function(x){
      table(x)/length(x)
    })
    res <- do.call("rbind", res)
    res <- melt(res)
    colnames(res) <- c("Set", "Cluster", "Proportion")
    res$Cluster <- as.character(res$Cluster)
  
    p <- ggplot(res, aes(x = Cluster, y = Set, fill = Proportion)) +
    geom_tile() +
    theme_classic() +
    scale_fill_viridis(option = "magma") +
    theme(text = element_text(face = "bold",
                              size = 15,
                              colour = "black"),
          line = element_line(colour = "black"),
          axis.text = element_text(colour = "black"),
          axis.text.x = element_blank()) +
    labs(title = "Contribution of clusters to each dataset",
         x = "",
         y = "")
  }
  return(p)
}

calculateClusterSetProp <- function(dat, setmatch){
  csm <- merge(setmatch, dat[,c("ID", "Cluster")], by = "ID", all.y = F)
  csm <- csm[!duplicated(csm$ID),]
  csm$Set <- as.factor(csm$Set)
  tmp <- split(csm$Set, as.factor(csm$Cluster))
  res <- lapply(tmp, function(x){
    table(x)/length(x)
  })
  res <- do.call("rbind", res)
  return(res)
}

calculateCorrelations <- function(Cluster, Dat, EM, Sets){
  Sub <- Dat[Dat$Cluster == as.character(Cluster),]
  Sub <- Sub[!duplicated(Sub$ID),]
  
  Sizes <- table(Sets[Sets$ID %in% Sub$ID, "Set"])
  TopSet <- names(Sizes[order(Sizes, decreasing = T)])[1]
    
  RestSets <- sapply(EM, `[`, as.character(Cluster))
  names(RestSets) <- gsub("\\..*", "", names(RestSets))
  Cors <- sapply(1:length(RestSets), function(x){
    cor(RestSets[[TopSet]], RestSets[[x]])
  })
  names(Cors) <- names(RestSets)
  Cors[-which(names(Cors) == TopSet)]
}

#Requires ggplot2, gridExtra
#phecodes_complete
#Expression matrices to be calculated beforehand
createPheSpec_multi <- function(N=1, Clusters, Dat, EM, BG, Sets, Tsne, Filter=NULL, Tops=NULL){
  #col_rb <- c(rainbow(length(unique(phecodes_complete$Category))), "lightgrey", "grey")
  col_rb <- c(viridis_pal(option = "D")(length(unique(phecodes_complete$Category))), "lightgrey")
  names(col_rb) <- c(unique(as.character(phecodes_complete$Category)), "Unknown")
  col_lab <- rep(c("white", "black"), times = c(10, 12))
  names(col_lab) <- c(unique(as.character(phecodes_complete$Category)), "Unknown")
  plot.new()
  legend(0.3,1, 
         legend = c(levels(factor(phecodes_complete$Category, 
                                levels = unique(phecodes_complete$Category))),
                    "Unknown"), 
         col = col_rb, 
         lty = 1,
         lwd = 2, 
         cex = 1)
         
  PlotBG <- data.frame(BG)
  PlotBG <- merge(BG, phecodes_complete[!duplicated(phecodes_complete$PheCode),c("PheCode", "Phenotype", "Category")], by.x = 1, by.y = "PheCode")
  colnames(PlotBG) <- c("Code", "Freq", "Label", "Cat")
  PlotBG$Code <- as.character(PlotBG$Code)
  PlotBG$Cat <- as.character(PlotBG$Cat)
  PlotBG[is.na(PlotBG$Cat), "Cat"] <- "Unknown"
	PlotBG$Label <- as.character(PlotBG$Label)
  PlotBG <- PlotBG[order(match(PlotBG$Cat, c(unique(as.character(phecodes_complete$Category)), "Unknown")), PlotBG$Code),]
  PlotBG$Cat <- factor(PlotBG$Cat, levels = c(unique(as.character(phecodes_complete$Category)), "Unknown"))
  PlotBG$Code <- factor(PlotBG$Code, levels = unique(PlotBG$Code))   
         
  pbg <- ggplot(PlotBG, aes(x = Code, y = Freq, fill = Cat, label = Label)) +
    geom_vline(xintercept = PlotBG[!duplicated(PlotBG$Cat), "Code"],
               linetype = "dashed",
               colour = "grey") +
    geom_bar(stat = "identity") +
		scale_fill_manual(values = col_rb) +
		theme_classic() +
		theme(axis.ticks.x = element_blank(),
			  axis.text.x = element_blank(),
			  legend.position = "none") +
		scale_y_continuous(expand = c(0,0),
                       limits = c(0,1.2)) +
		labs(title = paste0("Complete dataset"),
			 subtitle = paste0("N = ", nrow(Sets)),
			 x = "PheCode",
			 y = "Frequency") +
    geom_label_repel(data = subset(PlotBG, Freq >= sort(PlotBG$Freq, decreasing = T)[3]),
                     segment.size  = 0.2,
                     segment.color = "grey50",
                     segment.alpha = 0.5,
                     nudge_y = 0.2,
                     direction = "y",
                     size = 5,
                     aes(colour = Cat)) +
    scale_colour_manual(values = col_lab)
                     
  grid.arrange(grid.rect(gp=gpar(col="white")), grid.rect(gp=gpar(col="white")),
               pbg, grid.rect(gp=gpar(col="white")),
               grid.rect(gp=gpar(col="white")), grid.rect(gp=gpar(col="white")),
               nrow = 3, ncol = 2,
               heights = c(0.5,1,0.5),
               widths = c(1.9, 0.1))

  SetOrder <- table(Sets$Set)
  SetOrder <- SetOrder[order(SetOrder, decreasing = T)]
  SetOrder <- names(SetOrder)
         
  col_sets <- viridis_pal(option = "plasma")(length(unique(Sets$Set)))
  names(col_sets) <- SetOrder
  
  mytheme <- gridExtra::ttheme_default(
    core = list(fg_params=list(cex = 0.6)),
    colhead = list(fg_params=list(cex = 0.6)),
    rowhead = list(fg_params=list(cex = 0.6)))
    
  invisible({
    lapply(1:min(c(N, length(Clusters))), function(x){
    print(paste0("Creating PheSpec for Cluster ", Clusters[x]))
    Sub <- Dat[Dat$Cluster == as.character(Clusters[x]),]
    
    Size <- length(unique(Sub$ID))
    
    Sizes <- table(Sets[Sets$ID %in% Sub$ID, "Set"])
    
    Sub_u <- unique(Sub[,c("ID", "PheCode")])
    
    Comp <- table(Sub_u$PheCode)/Size
    
    PlotDF <- data.frame(Comp)
    PlotDF <- merge(PlotDF, phecodes_complete[!duplicated(phecodes_complete$PheCode),c("PheCode", "Phenotype", "Category")], by.x = 1, by.y = "PheCode", all.x = T)
    colnames(PlotDF) <- c("Code", "Freq", "Label", "Cat")
    PlotDF$Code <- as.character(PlotDF$Code)
    PlotDF$Cat <- as.character(PlotDF$Cat)
    PlotDF[is.na(PlotDF$Cat), "Cat"] <- "Unknown"
	  PlotDF$Label <- as.character(PlotDF$Label)
    PlotDF <- PlotDF[order(match(PlotDF$Cat, c(unique(as.character(phecodes_complete$Category)), "Unknown")), PlotDF$Code),]
    PlotDF$Cat <- factor(PlotDF$Cat, levels = c(unique(as.character(phecodes_complete$Category)), "Unknown"))
    PlotDF$Code <- factor(PlotDF$Code, levels = unique(PlotDF$Code))   
    
    if(!is.null(Filter)){
      Mat <- PlotDF[!PlotDF$Code %in% Filter,]
    } else {
      Mat <- PlotDF
    }
    if(is.null(Tops)){
      Mat <- Mat[order(Mat$Freq, decreasing = T), c("Code", "Freq", "Label")][1:10,]
    } else {
      Tops <- Tops[[Clusters[x]]][,1]
      Mat <- Mat[order(Mat$Freq, decreasing = T), c("Code", "Freq", "Label")]
      Mat <- Mat[Mat$Code %in% Tops,]
      Mat <- Mat[1:min(length(Tops),10),]
    }
    Mat <- Mat[,c("Code", "Freq", "Label")]
    Mat$Freq <- round(as.numeric(Mat$Freq), digits = 2)
    Mat$Label <- ifelse(nchar(Mat$Label) > 43, paste0(substring(Mat$Label, 1, 40), "..."), Mat$Label)
    Mat <- Mat[complete.cases(Mat),]
    m1 <- tableGrob(Mat, rows = NULL, theme = mytheme)
    
    included_codes <- PlotDF
    if(!is.null(Filter)) included_codes <- included_codes[!included_codes$Code %in% Filter,]
    included_codes <- included_codes[order(included_codes$Freq, decreasing = T),]
    included_codes <- included_codes[1:500, "Code"]
    
    PlotDF <- PlotDF[PlotDF$Code %in% included_codes,]
    
    Sets <- sapply(EM, `[`, as.character(Clusters[x]))
    names(Sets) <- gsub("\\..*", "", names(Sets))
    CorMain <- sapply(1:length(Sets), function(y){
      cor(Comp, Sets[[y]])
    })
    names(CorMain) <- names(Sets)
    
    SetLabels <- paste0(1:length(SetOrder), ". ", SetOrder, "\nSize = ", Sizes[SetOrder], " Cor = ", round(CorMain[SetOrder], 2))
    names(SetLabels) <- SetOrder

    SetPlot <- melt(do.call("rbind", Sets))
    colnames(SetPlot) <- c("Set", "Code", "Freq")
    SetPlot <- SetPlot[SetPlot$Code %in% included_codes,]
    SetPlot <- SetPlot[order(match(SetPlot$Code, levels(PlotDF$Code))),]
    SetPlot$Code <- factor(SetPlot$Code, levels = unique(SetPlot$Code))
    SetPlot <- SetPlot[order(match(SetPlot$Set, SetOrder)),]
    SetPlot$Set <- factor(SetPlot$Set, levels = unique(SetPlot$Set))
    
    Cors <- lapply(1:length(Sets), function(y){
      sapply(1:length(Sets), function(z){
        cor(Sets[[SetOrder[y]]], Sets[[SetOrder[z]]])
      })
    })
    Cors <- do.call("rbind", Cors)
    colnames(Cors) <- 1:length(SetOrder)
    rownames(Cors) <- 1:length(SetOrder)
    Cors <- melt(Cors)
    colnames(Cors) <- c("X", "Y", "Cor")
    Cors$X <- factor(Cors$X, levels = 12:1)
    Cors$Y <- factor(Cors$Y, levels = 12:1)
     
	  p1 <- ggplot(PlotDF, aes(x = Code, y = Freq, fill = Cat, label = Label)) +
    geom_vline(xintercept = PlotDF[!duplicated(PlotDF$Cat), "Code"],
               linetype = "dashed",
               colour = "grey") +
    geom_bar(stat = "identity") +
		scale_fill_manual(values = col_rb) +
		theme_classic() +
		theme(axis.ticks.x = element_blank(),
			  axis.text.x = element_blank(),
			  legend.position = "none") +
		scale_y_continuous(expand = c(0,0),
                       limits = c(0,1.2)) +
		labs(title = paste0("Complete Cluster ", Clusters[x]),
			 subtitle = paste0("N = ", Size),
			 x = "PheCode",
			 y = "Frequency") +
		geom_label_repel(data = subset(PlotDF, Freq >= sort(PlotDF$Freq, decreasing = T)[3]),
                     segment.size  = 0.2,
                     segment.color = "grey50",
                     segment.alpha = 0.5,
                     nudge_y = 0.2,
                     direction = "y",
                     size = 5,
                     aes(colour = Cat)) +
    scale_colour_manual(values = col_lab)
    
    p2 <- ggplot(SetPlot, aes(x = Code, y = Freq, fill = Set)) +
    geom_bar(stat = "identity") +
    facet_wrap(~Set, labeller = labeller(Set = SetLabels)) +
    scale_fill_manual(values = col_sets) +
    theme_classic() +
    theme(axis.ticks.x = element_blank(),
          axis.text.x = element_blank(),
          legend.position = "none") +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1.2)) +
    labs(title = paste0("Remaining Datasets Cluster ", Clusters[x]),
         x = "PheCode",
         y = "Frequency")
                 
    p3 <- ggplot(Cors, aes(x = X, y = Y, fill = Cor)) +
    geom_tile() +
    scale_fill_viridis(option = "inferno") +
    theme_classic() +
    theme(legend.key.size = unit(0.4, "cm"),
          axis.text = element_text(size = 5)) +
    labs(title = "Correlation plot",
         x = "",
         y = "")
         
    p4 <- highlightClusters(Tsne, Dat, Clusters[x], add.rug = T, add.legend = F)
                
	  grid.arrange(p1, m1,
                 p2, arrangeGrob(p3, p4, nrow = 2),
                 nrow = 2, ncol = 2,
                 heights = c(2,2),
                 widths = c(2,0.9))
    })
  })
}

createPheSpec_png <- function(N=1, Clusters, Dat, EM, BG, Sets, Tsne, Name=paste0(format(Sys.Date(), "%Y%d%m"), "_phespecs")){
  
  col_rb <- c(viridis_pal(option = "D")(length(unique(phecodes_complete$Category))), "lightgrey")
  names(col_rb) <- c(unique(as.character(phecodes_complete$Category)), "Unknown")
  png("tmp_phe_legend.png", width = 700)
  plot.new()
  legend(0.3,1, 
         legend = c(levels(factor(phecodes_complete$Category, 
                                levels = unique(phecodes_complete$Category))),
                    "Unknown"), 
         col = col_rb, 
         lty = 1,
         lwd = 2, 
         cex = 1)
  dev.off()
  
  PlotBG <- data.frame(BG)
  PlotBG <- merge(BG, phecodes_complete[!duplicated(phecodes_complete$PheCode),c("PheCode", "Phenotype", "Category")], by.x = 1, by.y = "PheCode")
  colnames(PlotBG) <- c("Code", "Freq", "Label", "Cat")
  PlotBG$Code <- as.character(PlotBG$Code)
  PlotBG$Cat <- as.character(PlotBG$Cat)
  PlotBG[is.na(PlotBG$Cat), "Cat"] <- "Unknown"
	PlotBG$Label <- as.character(PlotBG$Label)
  PlotBG <- PlotBG[order(match(PlotBG$Cat, c(unique(as.character(phecodes_complete$Category)), "Unknown")), PlotBG$Code),]
  PlotBG$Cat <- factor(PlotBG$Cat, levels = c(unique(as.character(phecodes_complete$Category)), "Unknown"))
  PlotBG$Code <- factor(PlotBG$Code, levels = unique(PlotBG$Code))   
         
  pbg <- ggplot(PlotBG, aes(x = Code, y = Freq, fill = Cat, label = Label)) +
    geom_vline(xintercept = PlotBG[!duplicated(PlotBG$Cat), "Code"],
               linetype = "dashed",
               colour = "grey") +
    geom_bar(stat = "identity") +
		scale_fill_manual(values = col_rb) +
		theme_classic() +
		theme(axis.ticks.x = element_blank(),
			  axis.text.x = element_blank(),
			  legend.position = "none") +
		scale_y_continuous(expand = c(0,0),
                       limits = c(0,1.2)) +
		labs(title = paste0("Complete dataset"),
			 subtitle = paste0("N = ", nrow(Sets)),
			 x = "PheCode",
			 y = "Frequency") +
    geom_label_repel(data = subset(PlotBG, Freq >= sort(PlotBG$Freq, decreasing = T)[3]),
                     segment.size  = 0.2,
                     segment.color = "grey50",
                     segment.alpha = 0.5,
                     nudge_y = 0.2,
                     direction = "y",
                     size = 5)
  png("tmp_phe_bg.png", width = 700)                   
  grid.arrange(grid.rect(gp=gpar(col="white")), grid.rect(gp=gpar(col="white")),
               pbg, grid.rect(gp=gpar(col="white")),
               grid.rect(gp=gpar(col="white")), grid.rect(gp=gpar(col="white")),
               nrow = 3, ncol = 2,
               heights = c(0.5,1,0.5),
               widths = c(1.9, 0.1))
  dev.off()
  
  SetOrder <- table(Sets$Set)
  SetOrder <- SetOrder[order(SetOrder, decreasing = T)]
  SetOrder <- names(SetOrder)
         
  col_sets <- viridis_pal(option = "plasma")(length(unique(Sets$Set)))
  names(col_sets) <- SetOrder
  
  mytheme <- gridExtra::ttheme_default(
    core = list(fg_params=list(cex = 0.6)),
    colhead = list(fg_params=list(cex = 0.6)),
    rowhead = list(fg_params=list(cex = 0.6)))
    
  invisible({
    lapply(1:min(c(N, length(Clusters))), function(x){
    print(paste0("Creating PheSpec for Cluster ", Clusters[x]))
    Sub <- Dat[Dat$Cluster == as.character(Clusters[x]),]
    
    Size <- length(unique(Sub$ID))
    
    Sizes <- table(Sets[Sets$ID %in% Sub$ID, "Set"])
    
    Sub_u <- unique(Sub[,c("ID", "PheCode")])
    
    Comp <- table(Sub_u$PheCode)/Size
    
    PlotDF <- data.frame(Comp)
    PlotDF <- merge(PlotDF, phecodes_complete[!duplicated(phecodes_complete$PheCode),c("PheCode", "Phenotype", "Category")], by.x = 1, by.y = "PheCode", all.x = T)
    colnames(PlotDF) <- c("Code", "Freq", "Label", "Cat")
    PlotDF$Code <- as.character(PlotDF$Code)
    PlotDF$Cat <- as.character(PlotDF$Cat)
    PlotDF[is.na(PlotDF$Cat), "Cat"] <- "Unknown"
	  PlotDF$Label <- as.character(PlotDF$Label)
    PlotDF <- PlotDF[order(match(PlotDF$Cat, c(unique(as.character(phecodes_complete$Category)), "Unknown")), PlotDF$Code),]
    PlotDF$Cat <- factor(PlotDF$Cat, levels = c(unique(as.character(phecodes_complete$Category)), "Unknown"))
    PlotDF$Code <- factor(PlotDF$Code, levels = unique(PlotDF$Code))   
    
    if(!is.null(Filter)){
      Mat <- PlotDF[!PlotDF$Code %in% Filter,]
    } else {
      Mat <- PlotDF
    }
    Mat <- Mat[order(Mat$Freq, decreasing = T), c("Code", "Freq", "Label")][1:10,]
    Mat <- Mat[,c("Code", "Freq", "Label")]
    Mat$Freq <- round(as.numeric(Mat$Freq), digits = 2)
    Mat$Label <- ifelse(nchar(Mat$Label) > 43, paste0(substring(Mat$Label, 1, 40), "..."), Mat$Label)
    m1 <- tableGrob(Mat, rows = NULL, theme = mytheme)
    
    included_codes <- PlotDF
    if(!is.null(Filter)) included_codes <- included_codes[!included_codes$Code %in% Filter,]
    included_codes <- included_codes[order(included_codes$Freq, decreasing = T),]
    included_codes <- included_codes[1:500, "Code"]
    
    PlotDF <- PlotDF[PlotDF$Code %in% included_codes,]
    
    Sets <- sapply(EM, `[`, as.character(Clusters[x]))
    names(Sets) <- gsub("\\..*", "", names(Sets))
    CorMain <- sapply(1:length(Sets), function(y){
      cor(Comp, Sets[[y]])
    })
    names(CorMain) <- names(Sets)
    
    SetLabels <- paste0(1:length(SetOrder), ". ", SetOrder, "\nSize = ", Sizes[SetOrder], " Cor = ", round(CorMain[SetOrder], 2))
    names(SetLabels) <- SetOrder

    SetPlot <- melt(do.call("rbind", Sets))
    colnames(SetPlot) <- c("Set", "Code", "Freq")
    SetPlot <- SetPlot[SetPlot$Code %in% included_codes,]
    SetPlot <- SetPlot[order(match(SetPlot$Code, levels(PlotDF$Code))),]
    SetPlot$Code <- factor(SetPlot$Code, levels = unique(SetPlot$Code))
    SetPlot <- SetPlot[order(match(SetPlot$Set, SetOrder)),]
    SetPlot$Set <- factor(SetPlot$Set, levels = unique(SetPlot$Set))
    
    Cors <- lapply(1:length(Sets), function(y){
      sapply(1:length(Sets), function(z){
        cor(Sets[[SetOrder[y]]], Sets[[SetOrder[z]]])
      })
    })
    Cors <- do.call("rbind", Cors)
    colnames(Cors) <- 1:length(SetOrder)
    rownames(Cors) <- 1:length(SetOrder)
    Cors <- melt(Cors)
    colnames(Cors) <- c("X", "Y", "Cor")
    Cors$X <- factor(Cors$X, levels = 12:1)
    Cors$Y <- factor(Cors$Y, levels = 12:1)
     
	  p1 <- ggplot(PlotDF, aes(x = Code, y = Freq, fill = Cat, label = Label)) +
    geom_vline(xintercept = PlotDF[!duplicated(PlotDF$Cat), "Code"],
               linetype = "dashed",
               colour = "grey") +
    geom_bar(stat = "identity") +
		scale_fill_manual(values = col_rb) +
		theme_classic() +
		theme(axis.ticks.x = element_blank(),
			  axis.text.x = element_blank(),
			  legend.position = "none") +
		scale_y_continuous(expand = c(0,0),
                       limits = c(0,1.2)) +
		labs(title = paste0("Complete Cluster ", Clusters[x]),
			 subtitle = paste0("N = ", Size),
			 x = "PheCode",
			 y = "Frequency") +
		geom_label_repel(data = subset(PlotDF, Freq >= sort(PlotDF$Freq, decreasing = T)[3]),
                     segment.size  = 0.2,
                     segment.color = "grey50",
                     segment.alpha = 0.5,
                     nudge_y = 0.2,
                     direction = "y",
                     size = 5)
      
    p2 <- ggplot(SetPlot, aes(x = Code, y = Freq, fill = Set)) +
    geom_bar(stat = "identity") +
    facet_wrap(~Set, labeller = labeller(Set = SetLabels)) +
    scale_fill_manual(values = col_sets) +
    theme_classic() +
    theme(axis.ticks.x = element_blank(),
          axis.text.x = element_blank(),
          legend.position = "none") +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1.2)) +
    labs(title = paste0("Remaining Datasets Cluster ", Clusters[x]),
         x = "PheCode",
         y = "Frequency")
                 
    p3 <- ggplot(Cors, aes(x = X, y = Y, fill = Cor)) +
    geom_tile() +
    scale_fill_viridis(option = "inferno") +
    theme_classic() +
    theme(legend.key.size = unit(0.4, "cm"),
          axis.text = element_text(size = 5)) +
    labs(title = "Correlation plot",
         x = "",
         y = "")
         
    p4 <- highlightClusters(Tsne, Dat, Clusters[x], add.rug = T, add.legend = F)
    
	png(paste0("tmp_phe_", Clusters[x], ".png"), width = 700)
	  grid.arrange(p1, m1,
                 p2, arrangeGrob(p3, p4, nrow = 2),
                 nrow = 2, ncol = 2,
                 heights = c(2,2),
                 widths = c(2,0.9))
	dev.off()
    })
  })
  plots <- lapply(c("legend", "bg", Clusters), function(y){
	rasterGrob(readPNG(paste0("tmp_phe_", y, ".png"), native = F), interpolate = F)
  })
  
  system("rm tmp_phe_*")
  
  pdf(paste0(Name, ".pdf"), width = 10)
  invisible(
  lapply(1:length(plots), function(z){
	do.call(grid.arrange, c(plots[z], nrow = 1, ncol = 1))
  })
  )
  dev.off()
}

highlightClusters <- function(tsne, clmatch, cl, add.rug=FALSE, add.legend=TRUE, size=0.1){
  clmatch <- clmatch[!duplicated(clmatch$ID), c("ID", "Cluster")]
  dat <- merge(tsne$Y, clmatch, by.x = "row.names", by.y = "ID")
  colnames(dat) <- c("ID", "X", "Y", "Cluster")	
	dat$Cluster <- as.character(dat$Cluster)
	dat[!dat$Cluster %in% cl, "Cluster"] <- "Other"
  dat <- dat[order(dat$Cluster, decreasing = T),]
	dat$Cluster <- as.factor(dat$Cluster)
	dat$Cluster <- relevel(dat$Cluster, ref = "Other")
	
  if(length(levels(dat$Cluster)) > 2){
    col_pal <- c("grey", rainbow(length(cl)))
  } else {
    col_pal <- c("grey", "purple4")
  }
	p <- ggplot(dat, aes(x = X, y = Y, colour = Cluster)) +
	geom_point(size = size) +
	theme_classic() +
  theme(axis.ticks = element_blank(),
        axis.text = element_blank()) +
	scale_colour_manual(values = col_pal) +
	labs(title = "Cluster position",
		   x = "",
	  	 y = "")
          
  if(add.rug) p <- p + 
                   geom_rug(data = dat[!dat$Cluster == "Other",], colour = "purple4", alpha = .2, outside = T) + 
                   coord_cartesian(clip = "off")
  if(!add.legend) p <- p + 
                       theme(legend.position = "none") 
  return(p)
}

constructTrajectory <- function(dat, clust=NA, set, setmatch, len=10, reverse=FALSE, plot=FALSE){
  dat <- dat[which(dat$Cluster == clust),]
  if(!missing(set) & !missing(setmatch)) dat <- dat[dat$ID %in% setmatch[setmatch$Set == set, "ID"],]
  if(reverse){
    dat <- dat[order(dat[,c("ID", "Age")], decreasing = T),]
  } else {
    dat <- dat[order(dat[,c("ID", "Age")], decreasing = F),]
  }
  upi <- unique(dat[,c("ID", "PheCode")])
  codes <- table(upi$PheCode)
  codes <- codes[order(codes, decreasing = T)][1:len]
  red <- upi[upi$PheCode %in% names(codes),]
  spl <- split(red$PheCode, as.factor(red$ID))

  res <- names(which.max(table(sapply(spl, head, 1))))
  prb <- max(table(sapply(spl, head, 1)))/length(spl)
  num <- max(table(sapply(spl, head, 1)))
  
  i <- 1
  for(i in 1:(len -1)){
  next.codes <- sapply(spl, function(x){
      if(res[length(res)] %in% x){
        cur.pos <- which(x == res[length(res)])
        if(cur.pos == length(x)) return(NA)
        if(cur.pos < len) return(x[[cur.pos + 1]])
      } else {
        return(NA)
      }
    })
  next.codes[next.codes %in% res] <- NA
  next.codes <- table(next.codes)
  res <- append(res, names(next.codes[order(next.codes, decreasing = T)][1]))
  prb <- append(prb, as.vector(next.codes[order(next.codes, decreasing = T)][1])/sum(next.codes))
  num <- append(num, sum(next.codes))
  i <- i + 1
  }

  pheno <- sapply(res, function(x) phecodes_complete[phecodes_complete$PheCode == as.character(x), "Phenotype"][1])
  mat <- data.frame(PheCode = res, 
                    Phenotype = pheno, 
                    N = num,
                    Proportion = round(num/length(spl), 3), 
                    Probability = round(prb, 3))
                    
  if(plot){
    ggplot(mat, aes(x = 1:10, y = Probability, size = Proportion, colour = Proportion, label = Phenotype)) +
	    geom_point() +
	    geom_line(size = 0.5,
			          colour = "black") +
      scale_colour_viridis(name = "Proportion of cluster\nwith this code",
                           guide = "legend") +
      scale_size_continuous(name = "Proportion of cluster\nwith this code",
                            range = c(1,15)) +
	    geom_label_repel(size = 3,
					             colour = "black",
                       segment.size  = 0.8,
                       segment.color = "grey50",
                       segment.alpha = 0.5) +
	    theme_classic() +
	    theme(axis.text.x = element_blank(),
		        axis.ticks.x = element_blank()) +
	    labs(title = "Trajectory plot",
		       x = "Order (First -> Last)",
		       y = "Transition Probability")
  } else {
  return(mat)
  }
}

calculateCodeFlow <- function(dat, clust=NA, set, setmatch, len=10, reverse=FALSE, singles=FALSE){
  dat <- dat[which(dat$Cluster == clust),]
  if(!missing(set) & !missing(setmatch)) dat <- dat[dat$ID %in% setmatch[setmatch$Set == set, "ID"],]
  if(missing(set)) set <- "Combined"
  if(reverse){
    dat <- dat[order(dat$ID, dat$Age, decreasing = T),]
  } else {
    dat <- dat[order(dat$ID, dat$Age, decreasing = F),]
  }
  upi <- dat[!duplicated(dat[,c("ID", "PheCode")]),]
  codes <- table(upi$PheCode)
  codes <- codes[order(codes, decreasing = T)][1:len]
  if(singles){
    red <- upi[upi$PheCode %in% names(codes),]
  } else {  
    red <- dat[dat$PheCode %in% names(codes),]
  }
  ages <- red[!duplicated(red$ID), "Age"]
  names(ages) <- unique(red$ID)
  red$AgeDif <- ceiling(red$Age - ages[as.character(red$ID)])
   
  sizes <- unique(red[,c("ID", "AgeDif")])
  sizes <- table(sizes$AgeDif)/length(unique(red$ID))
  sizes <- data.frame(sizes)
  colnames(sizes) <- c("Age", "Size")
  sizes$Age <- as.integer(as.character(sizes$Age))
  
  ba <- table(red$AgeDif, red$PheCode)
  ba <- t(scale(t(ba), center = F, scale = colSums(t(ba))))
  ba <- melt(ba)
  colnames(ba) <- c("Age", "Code", "Prop")
  ba <- merge(ba, phecodes_complete[!duplicated(phecodes_complete$PheCode) ,c("PheCode", "Phenotype")], by.x = "Code", by.y = "PheCode", all.x = T)
  ba[is.na(ba$Phenotype), "Phenotype"] <- ba[is.na(ba$Phenotype), "Code"]
  
  p <- ggplot(ba, aes(x = Age, y = Prop, fill = as.factor(Phenotype))) +
    geom_bar(stat = "identity",
             position = "stack") +
    scale_fill_viridis(option = "D",
                       name = "Phenotype",
                       discrete = T) +
    theme_classic() +
    scale_y_continuous(expand = c(0,0)) +
    labs(title = "Distribution top 10 PheCodes through time",
         subtitle = paste0("Cluster ", clust, " ", set, " N = ", length(unique(red$ID))),
         x = "Years since first top 10 code",
         y = "Proportion of top 10 codes") +
    annotate("smooth", 
             x = sizes$Age, 
             y = sizes$Size, 
             colour = "grey", 
             size = 2)
  
  return(p)
}

calculateCodePeaks <- function(dat, clust=NA, set, setmatch, len=10, reverse=FALSE, filter_codes=NULL){
  dat <- dat[which(dat$Cluster == clust),]
  if(!missing(set) & !missing(setmatch)) dat <- dat[dat$ID %in% setmatch[setmatch$Set == set, "ID"],]
  if(missing(set)) set <- "Combined"
  if(reverse){
    dat <- dat[order(dat$ID, dat$Age, decreasing = T),]
  } else {
    dat <- dat[order(dat$ID, dat$Age, decreasing = F),]
  }
  upi <- dat[!duplicated(dat[,c("ID", "PheCode")]),]
  codes <- table(upi$PheCode)
  if(!is.null(filter_codes)) codes <- codes[!names(codes) %in% filter_codes]
  codes <- codes[order(codes, decreasing = T)][1:len]
  red <- upi[upi$PheCode %in% names(codes),]
  ages <- red[!duplicated(red$ID), "Age"]
  names(ages) <- unique(red$ID)
  red$AgeDif <- ceiling(red$Age - ages[as.character(red$ID)])
   
  ba <- merge(red, phecodes_complete[!duplicated(phecodes_complete$PheCode) ,c("PheCode", "Phenotype")], by = "PheCode", all.x = T)
  ba[is.na(ba$Phenotype), "Phenotype"] <- ba[is.na(ba$Phenotype), "Code"]
  
  peaks <- lapply(unique(ba$Phenotype), function(x){
	  d <- density(ba[ba$Phenotype == x, "AgeDif"])
	  peak <- which.max(d$y)
	  X <- d$x[peak]
	  Y <- d$y[peak]
	  data.frame(Phenotype = x, X = X, Y = Y)
  })
  peaks <- do.call("rbind", peaks)
  peaks$Rank <- as.character(rank(peaks$X))
  
  ba <- merge(ba, peaks[,c("Phenotype", "Rank")], by = "Phenotype", all.x = T)
  ba$Phenotype <- paste0(ba$Rank, ". ", ba$Phenotype)
  ba <- ba[order(as.numeric(ba$Rank)),]
  
  peaks$Phenotype <- paste0(peaks$Rank, ". ", peaks$Phenotype)
  
  ba$Phenotype <- ifelse(nchar(ba$Phenotype) > 28, paste0(substring(ba$Phenotype, 1, 25), "..."), ba$Phenotype)
  ba$Phenotype <- factor(ba$Phenotype, levels = unique(ba$Phenotype))
  peaks$Phenotype <- ifelse(nchar(peaks$Phenotype) > 28, paste0(substring(peaks$Phenotype, 1, 25), "..."), peaks$Phenotype)
  
  p <- ggplot(ba, aes(x = AgeDif, colour = Phenotype)) +
    geom_point(shape = "", aes(y = 0)) +
    geom_density(size = 1.5,
                 show.legend = F) +
    scale_colour_discrete(name = "Phenotype") +
    theme_classic() +
    guides(colour = guide_legend(override.aes = list(size = 5, 
                                                     shape = 19))) +
    theme(text = element_text(face = "bold",
                              size = 15,
                              colour = "black"),
          line = element_line(colour = "black"),
          axis.text = element_text(colour = "black")) +
    labs(title = "Distribution top 10 PheCodes through time",
         subtitle = paste0("Cluster ", clust, " ", set, " N = ", length(unique(ba$ID))),
         x = "Years since first top 10 code",
         y = "Density")
  
  p <- p + geom_label(data = peaks, aes(x = X, y = Y, label = Rank, fontface = "bold"), show.legend = F)
  
  p <- p + coord_cartesian(xlim = c(0,max(peaks$X)+1))
  
  return(p)
}

investigateCodePrevalence <- function(Code, Dat, Set, Setmatch, Threshold=0.8, Top=10){
  mytheme <- gridExtra::ttheme_default(
  core = list(fg_params=list(cex = 0.6)),
  colhead = list(fg_params=list(cex = 0.6)),
  rowhead = list(fg_params=list(cex = 0.6)))
  
  if(!missing(Set) & !missing(Setmatch)) Dat <- Dat[Dat$ID %in% Setmatch[Setmatch$Set == Set, "ID"],]
  if(missing(Set)) Set <- "Combined"
      
  set_cols["Combined"] <- "purple4"
      
  TMP <- Dat[Dat$PheCode == Code,]
  TMP <- TMP[!duplicated(TMP$ID),]
  Prevs <- table(TMP$Cluster)/table(Dat[!duplicated(Dat$ID), "Cluster"])
  Plot <- data.frame(Prevs[order(Prevs, decreasing = T)])
  colnames(Plot) <- c("Cluster", "Prevalence")
  Plot <- Plot[complete.cases(Plot),]
  Threshold <- min(c(Threshold, Plot[10, "Prevalence"]))

  m <- tableGrob(Plot[Plot$Prevalence >= Threshold,], rows = NULL, theme = mytheme)
  
  p <- ggplot(Plot, aes(x = Cluster, y = Prevalence)) +
    geom_bar(stat = "identity",
             fill = set_cols[Set]) +
    geom_hline(yintercept = Threshold,
               colour = "grey",
               linetype = "dashed") +
    theme_classic() +
    scale_y_continuous(expand = c(0,0),
                       limits = c(0,1)) +
    theme(axis.text.x = element_blank(),
          axis.ticks.x = element_blank()) +
    labs(title = paste0("Prevalence of ", Code, " across ", Set,  " clusters"),
         subtitle = paste0("Average prevalence ", round(mean(Plot$Prevalence, na.rm = T), 2)),
         x = "Cluster",
         y = "Prevalence")
         
  grid.arrange(p,m,
               ncol = 2,
               widths = c(1.5, 0.5))    
}

createSubClustering <- function(dat, clust, harm, type=c("knn", "kmeans"), k=30, n=2){
	harm <- harm[rownames(harm) %in% dat[dat$Cluster == clust, "ID"],]
	
	if(type == "knn"){
    res <- Rphenograph(harm, k = k)

    print("") 
 
	  key <- data.frame(ID = rownames(harm)[as.numeric(res[[2]]$name)], SubCluster = paste0(clust, ".", res[[2]]$membership))
  }
  if(type == "kmeans"){
    res <- kmeans(harm, centers = n)	
    
    key <- data.frame(ID = rownames(harm), SubCluster = paste0(clust, ".", res$cluster))
  }
  
	return(key)
}

calculateSubEN <- function(dat, subkey){
  dat <- dat[dat$ID %in% subkey$ID,]
  
  spl <- split(dat$PheCode, as.factor(dat$ID))

  m <- sapply(spl, function(x){
    table(x)
  })

  m <- t(m)

  colnames(m) <- paste0("P", colnames(m))

  prep_glm <- sparse.model.matrix(~., as.data.frame(m))
  invisible(
  res <- lapply(unique(subkey$SubCluster), function(x){
    prep_en <- cbind(prep_glm, as.numeric(rownames(prep_glm) %in% subkey[subkey$SubCluster == x, "ID"]))
    colnames(prep_en)[ncol(prep_en)] <- "InCluster"

    train_control <- trainControl(method = "repeatedcv",
                                  number = 5,
                                  repeats = 5,
                                  search = "random",
                                  verboseIter = F)

    elastic_net_model <- train(InCluster ~.,
                               data = as.matrix(prep_en),
                               method = "glmnet",
                               preProcess = c("center", "scale"),
                               tuneLength = 25,
                               trControl = train_control)
                               
    coefs <- coef(elastic_net_model$finalModel, elastic_net_model$bestTune$lambda)
    tmp <- data.frame(Code = gsub("P", "", coefs@Dimnames[[1]][coefs@i + 1]), 
                      Beta = coefs@x)
    tmp <- tmp[-grep("Intercept", tmp[,1]),]
    tmp[order(tmp[,2], decreasing = T),][1:10,]
  })
  )
  names(res) <- unique(subkey$SubCluster)
  return(res)
}

	
visualiseSubClustering <- function(dat, subkey, full=FALSE){

  cols <- c(viridis_pal(option = "D")(length(unique(subkey$SubCluster))), "grey")
  names(cols) <- c(levels(subkey$SubCluster), "Other")

  if(!full){
   dat <- dat$Y[rownames(dat$Y) %in% subkey$ID,]
  } else {
   dat <- dat$Y
  }

  plot_tsne <- data.frame(X = dat[,1],
                          Y = dat[,2],
                          ID = rownames(dat)) 

  plot_tsne <- merge(plot_tsne, subkey, by.x = "ID", by.y = 1, all.x = T)
  colnames(plot_tsne)[ncol(plot_tsne)] <- "Cluster"
  
  plot_tsne$Cluster <- as.character(plot_tsne$Cluster)
  plot_tsne[is.na(plot_tsne$Cluster), "Cluster"] <- "Other"
  plot_tsne <- plot_tsne[order(plot_tsne$Cluster, decreasing = T),]

  plot_tsne$Cluster <- as.factor(plot_tsne$Cluster)
  if(full) plot_tsne$Cluster <- relevel(plot_tsne$Cluster, "Other")

  p <- ggplot(plot_tsne, aes(x = X, y = Y, colour = Cluster)) +
    geom_point(size = 2) +
    theme_classic() +
    theme(text = element_text(face = "bold",
                              size = 15,
                              colour = "black"),
          line = element_line(colour = "black"),
          axis.text = element_text(colour = "black")) +
    scale_colour_manual(values = cols) +
    theme(legend.position = "none") +
    labs(title = "tSNE embedding post-Harmony",
         subtitle = paste0("Showing kNN subclustering of cluster ", gsub("\\..*", "", subkey[1,2])),
         x = "",
         y = "")
         
  print(p)
}

createPrevalenceVolcano <- function(Code, EM, Set, UC, BG, Top=10){
  if(!missing(Set)){
    Dat <- EM[[Set]]
    Dat <- Dat[-which(is.na(names(Dat)))]
  } else {
    Dat <- lapply(levels(UC$Cluster), function(x){
      table(UC[UC$Cluster == x, "PheCode"])/length(unique(UC[UC$Cluster == x, "ID"]))
    })
    names(Dat) <- levels(UC$Cluster)
  }
  if(missing(Set)) Set <- "Combined"
      
  set_cols["Combined"] <- "purple4"
  
  res <- data.frame()
  
  ra_sub <<- Dat[["18"]]
  
  invisible(   
  code_ranks <- sapply(1:length(Dat), function(x){
    tmp <- as.vector(Dat[[x]])
    names(tmp) <- names(Dat[[x]])
    res[x, "Cluster"] <<- names(Dat)[x]
    res[x, "Prev"] <<- tmp[[Code]]
    res[x, "Rank"] <<- rank(-tmp, ties.method = "min")[[Code]]
  })
  )
  
  res$Rank <- ifelse(res$Rank > Top, paste0(">", Top), res$Rank)
  
  res$Rank <- factor(res$Rank, levels = c(paste0(">", Top), Top:1))
  
  if(length(which(!is.finite(res$Prev))) > 0) res <- res[-which(!is.finite(res$Prev)),]
  
  p <- ggplot(res, aes(x = Rank, y = Prev, label = Cluster)) +
    geom_point(colour = set_cols[[Set]]) +
    geom_hline(yintercept = BG[[Code]],
               colour = "grey",
               linetype = "dashed") +
    scale_x_discrete(breaks = levels(res$Rank)) +
    scale_y_continuous(breaks = seq(0,1,0.25),
                       limits = c(0, 1)) +
    geom_label_repel(data = subset(res, Prev >= BG[[Code]] & Rank %in% levels(res$Rank)[2:length(levels(res$Rank))]),
                     segment.size  = 0.2,
                     segment.color = "grey50",
                     segment.alpha = 0.5,
                     size = 5,
                     colour = set_cols[[Set]],
                     fontface = "bold") +
    theme_classic() +
    theme(legend.position = "none",
          text = element_text(face = "bold",
                              size = 15,
                              colour = "black"),
          line = element_line(colour = "black"),
          axis.text = element_text(colour = "black")) +
    labs(title = paste0("Cluster of interest for code ", Code),
         subtitle = paste0(Set, " Data"),
         x = "Rank within cluster",
         y = "Prevalence within cluster")
    
  print(p)
}

extractCenters <- function(tsne, dat){
  coords <- lapply(levels(dat$Cluster), function(x){
    tmp <- tsne$Y[rownames(tsne$Y) %in% dat[dat$Cluster == x, "ID"],]
    res <- c(mean(tmp[,1]), mean(tmp[,2]), nrow(tmp))
    names(res) <- c("X", "Y", "Size")
    return(res)
  })
  coords <- do.call("rbind", coords)
  rownames(coords) <- levels(dat$Cluster)
  return(as.data.frame(coords))
}