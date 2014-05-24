# setwd("Kaggle/NCAA/R")

library("dplyr")
library("caret")
library("gbm")

source("../../../PWMisc/R/UtilitySources.r")
source("../../../PWMisc/R/Graphics.r")

FileDir <- "../data"

###############################################################################
#
# algo_preds()-
#
###############################################################################

algo_preds <- function(FileDir="../data"){
  
  start_day <- 133
  first_season <- "H"
  
  #Pull the data
  
  # tourney_slots <- read.csv(paste0(FileDir, "/tourney_slots.csv"))
  #	ordinal_ranks_non_core <- read.csv(paste0(FileDir, "/ordinal_ranks_non_core.csv")) #season H
  #	regular_season_results <- read.csv(paste0(FileDir, "/regular_season_results.csv"))
  #	pointspreads <- read.csv(paste0(FileDir, "/pointspreas.csv")) #season L
  
  submissions_df <- read.csv(paste0(FileDir, "/sample_submission.csv"))
  names(submissions_df) <- c("id", "outcome")
  submissions_df$outcome <- NA
  
  
  seasons <- read.csv(paste0(FileDir, "/seasons.csv"))
  teams <- read.csv(paste0(FileDir, "/teams.csv"))
  
  sagp_weekly_ratings <- read.csv(paste0(FileDir, "/sagp_weekly_ratings.csv"))
  #season H
  ordinal_ranks_core_33 <- read.csv(paste0(FileDir, "/ordinal_ranks_core_33.csv"))
  ordinal_ranks_season_S <- read.csv(paste0(FileDir, "/ordinal_ranks_season_S.csv"))
  
  ordinal_ranks_season_S$rating_day_num[ordinal_ranks_season_S$rating_day_num == 128] <- start_day
  
  ordinal_ranks_core_33 <- rbind(ordinal_ranks_core_33, ordinal_ranks_season_S)
  
  #season H
  tourney_seeds <- read.csv(paste0(FileDir, "/tourney_seeds.csv"))
  tourney_results <- read.csv(paste0(FileDir, "/tourney_results.csv"))
  
  yearly_statistics <- read.csv(paste0(FileDir, "/yearly_statisics.csv"))
  
  #
  #Create the response dataframe
  #
  
  tourney_response <- tourney_results
  tourney_response$lower_id <- pmin(tourney_response$wteam, tourney_response$lteam)
  tourney_response$higher_id <-pmax(tourney_response$wteam, tourney_response$lteam)
  
  tourney_response$id <- paste(tourney_response$season,
    tourney_response$lower_id, tourney_response$higher_id, sep="_")
  
  tourney_response$outcome <- 1
  tourney_response$outcome[tourney_response$wteam == tourney_response$higher_id] <- 0

  tourney_response <- tourney_response[tourney_response$season >= first_season,
                                       c("id", "outcome")]
  
  #
  #Clean and process data
  #
  
  tourney_seeds$seed <- as.integer(gsub("[^0-9]", "", tourney_seeds$seed))
  ordinal_weekly_ranks <- ordinal_ranks_core_33
  #	ordinal_weekly_ranks <- rbind(ordinal_ranks_core_33, ordinal_ranks_non_core)
  #Rename the 7OT system to SVT
  ordinal_weekly_ranks$sys_name[ordinal_weekly_ranks$sys_name=="7OT"] <- "SVT"
  #Remove all ranks that are just 1-25 rankings or that don't have data for day
  full_rank_systems <- ddply(ordinal_weekly_ranks, "sys_name", summarize,
                             max_rank=max(orank),
                             tourney_rank=sum(rating_day_num==133) > 0, first_season=min(season))
  full_rank_systems <- full_rank_systems$sys_name[full_rank_systems$max_rank > 100 &
                                                  full_rank_systems$tourney_rank & full_rank_systems$first_season=="H"]
  ordinal_weekly_ranks <- ordinal_weekly_ranks[
    is.element(ordinal_weekly_ranks$sys_name, full_rank_systems), ]
  
  
  #Convert to a rating system
  ordinal_weekly_ratings <- ordinal_weekly_ranks
  ordinal_weekly_ratings$rating <- rank_to_rating(ordinal_weekly_ratings$orank)
  
  #Create the rating dataframe (for right before the tourney starts)
  
  #Sagaring ratings
  sagp_ratings <- sagp_weekly_ratings[sagp_weekly_ratings$rating_day_num == start_day,
                                      c("season", "team", "rating")]
  names(sagp_ratings) [3] <- "SAGP"
  
  #All ranks
  ordinal_ratings <-
    dcast(ordinal_weekly_ratings[ordinal_weekly_ratings$rating_day_num == start_day, ],
          season + team ~ sys_name, value.var="rating")
  
  ratings_df <- merge(sagp_ratings, ordinal_ratings, by=c("season", "team"), all=TRUE)
  ratings_df$SAGP[is.na(ratings_df$SAGP)] <- ratings_df$SAG[is.na(ratings_df$SAGP)]
  
  
  #Add the S season to the results
  tourney_response <- rbind(tourney_response, submissions_df)
  
  #
  #Create the explanatory dataframe
  #Including feature creation
  #
  col_info <- 5
  
  #Assemble the features matrix
  features_lst <- create_features(tourney_response, ratings_df, tourney_seeds, yearly_statistics)
  prediction_df <- features_lst$prediction_df
  ratings_spread <- features_lst$ratings_spread
  
  #
  #
  #Modeling
  #
  #
  
  #Create data partitions
  set.seed(1417)
#   ndxTrain <- createDataPartition(prediction_df$outcome, p=.7, list=FALSE)
  ndxTrain <- which(prediction_df$season != "S")
  
  dat_train <- prediction_df[ndxTrain, ]
  dat_test <- prediction_df[-ndxTrain, ]
  
  #Import loss function and cross validation
  tr_control <- trainControl(method="cv", number=10,
                             summaryFunction=loglossprob_summary,
                             predictionBounds=c(0,1))

  preProc <- preProcess(dat_train[, -c(1:col_info)], method="bagImpute")
  dat_train[, -c(1:col_info)] <- predict(preProc, dat_train[, -c(1:col_info)])
  
  
  #Testing framework
  run_model <- function(feature_columns=(col_info+1):ncol(dat_train), method="glm"){
  
    model <- train(x=dat_train[, feature_columns, drop=FALSE],
                  y=dat_train$outcome,
                  method=method,
                  family=binomial(link="logit"),
                  trControl=tr_control,
                  metric="logloss",
                  maximize=FALSE)

    preds <- data.frame(dat_test[, 1:col_info], predicted=
                        predict(model, newdata=dat_test[, feature_columns, drop=FALSE]))
                  
    output <- preds[ , c("outcome", "predicted")]
    names(output) <- c("obs", "pred")
    
    return(output)
  }
                      
  test_model_cv <- function(feature_columns=(col_info+1):ncol(dat_train), method="glm"){
    
    model <- train(x=prediction_df[, feature_columns, drop=FALSE],
                    y=prediction_df$outcome,
                    method=method,
                    family=binomial(link="logit"),
                    trControl=tr_control,
                    metric="logloss",
                    maximize=FALSE)
    
    preds <- data.frame(dat_test[, 1:col_info], predicted=
                        predict(model, newdata=dat_test[, feature_columns, drop=FALSE]))
                      
    return(model$results$logloss)
  }
                      
  #
  #Exploration
  #
  
#   system_scores_cv <- sapply(c((col_info+1):(ncol(dat_train)-3)), test_model_cv)
#   names(system_scores_cv) <- names(dat_train)[(col_info+1):(ncol(dat_train)-3)]
#   system_preds <- lapply(c((col_info+1):ncol(dat_train)), run_model)
#   names(system_preds) <- names(dat_train)[(col_info+1):ncol(dat_train)]
#   system_scores_test <- sapply(system_preds, loglossprob_summary)
#   names(system_scores_test) <- names(dat_train)[(col_info+1):ncol(dat_train)]
                      
                      
  #Sample models
  

  grid_cv <- data.frame(interaction.depth=2, n.trees=seq(1000,7000,10000), shrinkage=.001)

  model_all_gbm <- train(x=dat_train[, -c(1:col_info), drop=FALSE],
                         y=dat_train$outcome,
                         method="gbm",
                         trControl=tr_control,
                         tuneGrid=grid_cv,
                         metric="logloss",
                         maximize=FALSE)
  
  preds_all_gbm <- data.frame(dat_test[, 1:col_info], predicted=
                                predict(model_all_gbm, newdata=dat_test[, -c(1:col_info), drop=FALSE]))
  evaluation_all_gbm <- preds_all_gbm[ , c("outcome", "predicted")]
  names(evaluation_all_gbm) <- c("obs", "pred")
  loglossprob_summary(evaluation_all_gbm)
  
  
  # Top 5 scoring systems
#   top5_sys <- names(head(sort(system_scores_cv), 5))
#   
#   model_top5_gbm <- train(x=dat_train[, top5_sys, drop=FALSE],
#                           y=dat_train$outcome,
#                           method="gbm",
#                           trControl=tr_control,
#                           tuneGrid=grid_cv,
#                           metric="logloss",
#                           maximize=FALSE)
#   preds_top5_gbm <- data.frame(dat_test[, 1:col_info], predicted=
#                                predict(model_top5_gbm, newdata=dat_test[, top5_sys, drop=FALSE]))
#   evaluation_top5_gbm <- preds_top5_gbm[ , c("outcome", "predicted")]
#   names(evaluation_top5_gbm) <- c("obs" , "pred")
#   loglossprob_summary(evaluation_top5_gbm)
  
  
  #
  #Submissions and error analysis
  #
#   browser()
# 
#   features_lst_test <- create_features(submissions_df, ratings_df, tourney_seeds, yearly_statistics)
#   prediction_df_subm <- features_lst_test$prediction_df
#   
#   
#   preds_all_gbm_subm <- data.frame(id=prediction_df_subm$id, pred=
#     predict(model_all_gbm, newdata=prediction_df_subm[, all_systems, drop=FALSE]))
#   
#   seeds_subm <- data.frame(id=prediction_df_subm$id, pred=prediction_df_subm$SEED)
#   
#   
#   #Local scoring
#   seeds_subm_test <- na.omit(data.frame(
#                               id=tourney_response$id,
#                               obs=tourney_response$outcome,
#                               pred=seeds_subm$pred[match(tourney_response$id,
#                                                    seeds_subm$id)]))
#   
#   preds_top5_glm_subm_test <- na.omit(data.frame(
#                               id=tourney_response$id,
#                               obs=tourney_response$outcome,
#                               pred=preds_top5_glm_subm$pred[match(tourney_response$id,
#                                                             preds_top5_glm_subm$id)]))
#   
#   #Weighting more multiple models
#   combine_preds <- function(lst, wghts){
#     return(Reduce("+", Map("*", lapply(lst, function(x) x$pred), wghts)))
#   }
#   
#   #Weighted Top 5 and Seeds
#   wghts <- c(.875, .125)
#   combined_subm_test <- data.frame(preds_top5_glm_subm_test[, -3], pred=
#       combine_preds(lst=list(preds_top5_glm_subm_test, seeds_subm_test), wghts=wghts))
#   
#   #Weighted Combined and 0.5 for games that are hard to predict
#   combined_subm_close_games_test <- combined_subm_test
#   combined_subm_close_games_test$pred[abs(combined_subm_close_games_test$pred - .5) < .08] <- .5
#   
#   print("Seeds score")
#   print(loglossprob_summary(seeds_subm_test))
#   
#   print("Top 5 score")
#   print(loglossprob_summary(preds_top5_glm_subm_test))
#   
#   print("Combined score")
#   print(loglossprob_summary(combined_subm_test))
#   
#   print("Combined score with close games")
#   print(loglossprob_summary(combined_subm_close_games_test))
#   
#   #Error analysis
#   
#   seeds_errors <- logloss_share(seeds_subm_test)
#   seeds_errors <- seeds_errors[, c("id", "logloss", "loglossshare")]
#   names(seeds_errors) <- c("id", "logloss_seeds", "loglossshare_seeds")
#   top5_errors <- logloss_share(combined_subm_close_games_test)
#   
#   error_df <- merge(prediction_df_subm, top5_errors, by="id", all=FALSE)
#   error_df <- merge(error_df[, -5], seeds_errors, by="id", all=FALSE)
#   
#   error_df <- error_df[order(error_df$logloss), ]
#   
#   error_df$logloss_buckets <- cut(error_df$logloss, 20)
#   error_df$correct <- ifelse(round(error_df$pred)==error_df$obs, TRUE, FALSE)
#   error_df$pred_buckets <- cut(abs(.5 - error_df$pred), 10)
#   error_df$pred_buckets_seed <- cut(abs(.5 - error_df$SEED), 10)
#   
#   error_buckets <- dcast(
#     ddply(error_df, c("correct", "pred_buckets"), summarize,
#           total_share=sum(loglossshare)),
#     pred_buckets ~ correct, value.var="total_share"
#   )
#   
#   error_buckets_count <- dcast(
#     ddply(error_df, c("correct", "pred_buckets"), summarize,
#           total_share=length(loglossshare)),
#     pred_buckets ~ correct, value.var="total_share"
#   )
#   
#   error_buckets_seed <- dcast(
#     ddply(error_df, c("correct", "pred_buckets_seed"), summarize,
#           total_share=sum(loglossshare)),
#     pred_buckets_seed ~ correct, value.var="total_share"
#   )
#   
#   error_buckets_count_seed <- dcast(
#     ddply(error_df, c("correct", "pred_buckets_seed"), summarize,
#           total_share=length(loglossshare)),
#     pred_buckets_seed ~ correct, value.var="total_share"
#   )
#   
#   print("Error buckets")
#   print(error_buckets)
#   print("Error buckets count")
#   print(error_buckets_count)
#   
#   print("Error buckets - seed")
#   print(error_buckets_seed)
#   print("Error buckets count - seed")
#   print(error_buckets_count_seed)
  
  #
  #   #Write files
  #

  browser()

  write.csv(preds_all_gbm,
    paste0(FileDir, "/submissions/submission_algo ", Sys.Date(), ".csv"),
    row.names=FALSE)

  #Convert to matrix
  matrix_preds <- preds_all_gbm[, c("team1", "team2", "id", "predicted")]

  matrix_preds$team1 <- teams$name[match(matrix_preds$team1, teams$id)]
  matrix_preds$team2 <- teams$name[match(matrix_preds$team2, teams$id)]

  matrix_preds_rev <- matrix_preds
  matrix_preds_rev$team1 <- matrix_preds$team2
  matrix_preds_rev$team2 <- matrix_preds$team1
  matrix_preds_rev$predicted <- 1 - matrix_preds$predicted

  dd <- dcast(rbind(matrix_preds, matrix_preds_rev),
    team2 ~ team1, value.var="predicted")
  matrix_output <- as.matrix(dd[, -1])
  rownames(matrix_output) <- dd[, 1]

  #Predicted winner on top, loser on left
  write.csv(matrix_output, paste0(FileDir, "/submissions/predict_matrix_winner_top", Sys.Date(), ".csv"))
}

###############################################################################
#
# create_features()-
#
###############################################################################

create_features <- function(response, ratings_df, seeds, yearly_statistics){
  
  response_df <- cbind(response, ldply(response$id, function(x) t(data.frame(strsplit(x, split="_")))))
  names(response_df) <- c("id", "outcome", "season", "team1", "team2")
  
  #Add in the ranks
  col_start <- ncol(response_df) + 1
  pred_df <- merge(response_df, ratings_df, by.x=c("season", "team1"), by.y=c("season", "team"))
  names(pred_df)[col_start:ncol(pred_df)] <-paste0(names(pred_df)[col_start:ncol(pred_df)], "_team1")
  
  col_start <- ncol(pred_df) + 1
  pred_df <- merge(pred_df, ratings_df, by.x=c("season", "team2"), by.y=c("season", "team"))
  names(pred_df)[col_start:ncol(pred_df)] <- paste0(names(pred_df)[col_start:ncol(pred_df)], "_team2")
  
  #Calculate a ratings spread
  # id | outcome | season | team | team | ...
  col_info <- sum(is.element(c("season", "team2", "team1", "id", "outcome"), names(pred_df)))
  
  vars_sys <- ldply(c((col_info + 1):ncol(pred_df)), function(ccc){
    nm_split <- strsplit(names(pred_df)[ccc], "_")[[1]]
    data.frame(var=nm_split[1], team=nm_split[2], col=ccc)
  })
  
  vars_sys <- dcast(vars_sys, var~team, value.var="col")
  if(!all.equal(names(vars_sys), c("var", "team1", "team2"))) stop()
  
  ratings_spread_lst <- apply(vars_sys, 1, function(var){
    dat <- data.frame(new=pred_df[, as.integer(var[2])] - pred_df[, as.integer(var[3])])
    names(dat) <- var[1]
    return(dat)
  })
  
  ratings_spread_t <- t(Reduce(cbind, ratings_spread_lst))
  
  #Impute missing values
  ratings_spread <- apply(ratings_spread_t, 1, function(x){
    mn <- mean(x, na.rm=TRUE)
    x[is.na(x)] <- mn
    return(x)
  })
  
  #Turn rating spread into a predictor (win percentage)
  ratings_winpct <- rating_spread_to_win_pct(ratings_spread)
  
  #Recombine the ratings into the prediction df
  prediction_df <- cbind(pred_df[, 1:col_info], ratings_winpct)
  
  #Add in the seeds
  tourney_seeds1 <- seeds
  names(tourney_seeds1) <- c("season", "seed1", "team1")
  tourney_seeds2 <- seeds
  names(tourney_seeds2) <- c("season", "seed2", "team2")
  
  prediction_df <- merge(prediction_df, tourney_seeds1, by=c("season", "team1"))
  prediction_df <- merge(prediction_df, tourney_seeds2, by=c("season", "team2"))
  
  #Naive seed algorithm
  prediction_df$SEED <- seeds_to_win_pct(prediction_df$seed2, prediction_df$seed1)
  
  
  #Merge in the yearly stats to calculate stat differences
  yearly_stats1 <- yearly_statistics
  names(yearly_stats1) <- c("team1", "season", "RPG_team1", "OPPRPG_team1", "REB.MAR_team1", "TOPG_team1", "TO.MAR_team1")
  yearly_stats2 <- yearly_statistics
  names(yearly_stats2) <- c("team2", "season", "RPG_team2", "OPPRPG_team2", "REB.MAR_team2", "TOPG_team2", "TO.MAR_team2")
  
  statsDF <- merge(pred_df[, 1:col_info], yearly_stats1, by=c("season", "team1"))
  statsDF <- merge(statsDF, yearly_stats2, by=c("season", "team2"))
  
  statsDF$RPGdiff <- statsDF$RPG_team1 - statsDF$RPG_team2
  statsDF$OPPRPGdiff <- statsDF$OPPRPG_team1 - statsDF$OPPRPG_team2
  statsDF$REB.MARdiff <- statsDF$REB.MAR_team1 - statsDF$REB.MAR_team2
  statsDF$TOPGdiff <- statsDF$TOPG_team1 - statsDF$TOPG_team2
  statsDF$TO.MARdiff <- statsDF$TO.MAR_team1 - statsDF$TO.MAR_team2
  
  prediction_df <- merge(prediction_df, statsDF[, c("team1", "team2", "season", "RPGdiff",
                                                    "OPPRPGdiff", "REB.MARdiff", "TOPGdiff", "TO.MARdiff")], 
                         by=c("season", "team1", "team2"))  
  
  return(list(prediction_df=prediction_df, ratings_spread=cbind(response_df, ratings_spread)))
}

###############################################################################
#
# loglossprob_summary()-
#
###############################################################################
loglossprob_summary <- function(data, lev = NULL, model = NULL){
  y <- data$obs
  yhat <- data$pred
  
  yhat[yhat < 0] <- 0
  yhat[yhat > 1] <- 1
  
  out <- -sum(y * log(yhat) + (1-y)*log(1-yhat)) / length(y)
  names(out) <- c("logloss")
  return(out)
}

###############################################################################
#
# logloss_share()-
#
###############################################################################

logloss_share <- function(data){
  y <- data$obs
  yhat <- data$pred
  
  yhat[yhat < 0] <- 0
  yhat[yhat > 1] <- 1
  data$logloss <- y * log(yhat) + (1-y)*log(1-yhat)
  data$loglossshare <- data$logloss / sum(data$logloss)
  
  return(data)
}

###############################################################################
#
# rank_to_rating()-
#
###############################################################################

rank_to_rating <- function(rank){
  return(100 - 4 * log(rank+1) - rank /22)
}

###############################################################################
#
# rating_spread_to_win_pct()-
#
###############################################################################

rating_spread_to_win_pct <- function(rating_spread){
  return(1 / (1 + exp(-rating_spread / 14.5)))
}

###############################################################################
#
# seeds_to_win_pct()-
#
###############################################################################

seeds_to_win_pct <- function(high, low){
  return(0.50 + 0.0333 * (high - low))
}

###############################################################################
#
# manipulate_preds()-
#
###############################################################################

manipulate_preds <- function(){
  
  preds <- read.csv(paste0(FileDir, "/submissions/submission_algo ", Sys.Date(), ".csv"))
  preds <- read.csv(paste0(FileDir, "/submissions/submission_algo 2014-03-18.csv"))
  
  teams <- read.csv(paste0(FileDir, "/teams.csv"))
  injuries <- read.csv(paste0(FileDir, "/injuries.csv"))
  
  names(injuries)[1] <- "name"

  injuryDat <- merge(injuries, teams, by="name")
  injuryDat <- injuryDat[, c("id", "pool1", "pool2")]
  names(injuryDat)[1] <- "team"
  
  preds_1 <- merge(preds, injuryDat[, c("team", "pool1")], by.x="team1", by.y="team", all.x=TRUE)
  names(preds_1)[7] <- "team1boost"
  preds_1 <- merge(preds_1, injuryDat[, c("team", "pool1")], by.x="team2", by.y="team", all.x=TRUE)
  names(preds_1)[8] <- "team2boost"
  
  preds_1$predicted <- preds_1$predicted + preds_1$team1boost - preds_1$team2boost
  
  preds_1 <- preds_1[, c("id", "predicted")]
  names(preds_1) <- c("id", "pred")
  
  preds_2 <- merge(preds, injuryDat[, c("team", "pool2")], by.x="team1", by.y="team", all.x=TRUE)
  names(preds_2)[7] <- "team1boost"
  preds_2 <- merge(preds_2, injuryDat[, c("team", "pool2")], by.x="team2", by.y="team", all.x=TRUE)
  names(preds_2)[8] <- "team2boost"
  
  preds_2$predicted <- preds_2$predicted + preds_2$team1boost - preds_2$team2boost
  
  preds_2 <- preds_2[, c("id", "predicted")]
  names(preds_2) <- c("id", "pred")
  
  preds_1$pred[preds_1$pred < .00001] <- .00001
  preds_1$pred[preds_1$pred > .99999] <- .99999
  
  preds_2$pred[preds_2$pred < .00001] <- .00001
  preds_2$pred[preds_2$pred > .99999] <- .99999
  
  write.csv(preds_1, paste0(FileDir, "/submissions/preds_1_ ", Sys.Date(), ".csv"),
            row.names=FALSE)
  write.csv(preds_2, paste0(FileDir, "/submissions/preds_2_ ", Sys.Date(), ".csv"),
            row.names=FALSE)
}

###############################################################################
#
# cleanStatisticsData()-
#
###############################################################################

cleanStatisticsData <- function(){
  
  firstSeason <- 2003
  
  seasonsID <- c("H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S")
  
  allRebounds <- NULL
  allTurnovers <- NULL
  allTurnoverMargin <- NULL
  
  for(sss in 1:length(seasonsID)){
    
    year <- firstSeason + sss - 1
    
    season <- seasonsID[sss]
    
    rebounds <- read.csv(paste0(FileDir, "/SeasonData/", year, " Rebounds.csv"))
    rebounds$season <- season 
    allRebounds <- rbind(allRebounds, rebounds)                     
    
    
    turnovers <- read.csv(paste0(FileDir, "/SeasonData/", year, " Turnovers.csv"))
    turnovers$season <- season 
    allTurnovers <- rbind(allTurnovers, turnovers)                     
    
    if(file.exists(paste0(FileDir, "/SeasonData/", year, " Turnover Margin.csv"))){
    
      turnoverMargin <- read.csv(paste0(FileDir, "/SeasonData/", year, " Turnover Margin.csv"))
      turnoverMargin$season <- season 
      allTurnoverMargin <- rbind(allTurnoverMargin, turnoverMargin)                     
    }
  }
  
  teams <- read.csv(paste0(FileDir, "/teams.csv"))
  
  allDat <- merge(allRebounds, allTurnovers, by=c("Name", "season"), all=TRUE)
  allDat <- merge(allDat, allTurnoverMargin, by=c("Name", "season"), all=TRUE)
  allDat <- allDat[allDat$Name != "", ]
  
  allDat$Name <- gsub("\\.", "", allDat$Name)
  allDat$Name <- gsub("\\(", "", allDat$Name)
  allDat$Name <- gsub("\\)", "", allDat$Name)
  
  teamNames <- unique(allDat$Name)
  missingNames <- sort(teamNames[!is.element(teamNames, teams$name)])
  
  mapping <- read.csv(paste0(FileDir, "/SeasonData/missing names.csv"))
  
  allDat$Name2 <- mapping$new[match(allDat$Name, mapping$old)]
  allDat$Name[!is.na(allDat$Name2)] <- allDat$Name2[!is.na(allDat$Name2)]
  
  teamNames <- unique(allDat$Name)
  missingNames <- sort(teamNames[!is.element(teamNames, teams$name)])
  
  
  allDat <- allDat[!is.element(allDat$Name, missingNames), c(1:7)]
  names(allDat)[1] <- "name"
  
  allDat <- merge(allDat, teams, by="name", all.x=TRUE)
  
  allDat <- allDat[, c("id", "season", "RPG", "OPP.RPG", "REB.MAR", "TOPG", "Turnover.Margin")]
  names(allDat) <- c("id", "season", "RPG", "OPP_RPG", "REB.MAR", "TOPG", "TO.MAR")
  
  write.csv(allDat, paste0(FileDir, "/yearly_statisics.csv"), row.names=FALSE)
}
