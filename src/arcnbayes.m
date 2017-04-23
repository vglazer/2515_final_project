function tcr = nbayes(data, thresh, nrounds, perfold, allinfo, nfolds)

if nargin < 6
   nfolds = 10;
end 
if nargin < 5
   allinfo = false;
end
if nargin < 4
   perfold = false;
end
if nargin < 3 
   nrounds = 15;
end
if nargin < 2
   thresh = 0.5;
end

nfeats = size(data,1);
fsize = size(data,2)/nfolds;
charac = data(nfeats,:);
data = data(1:nfeats-1,:);
lambda = thresh/(1-thresh);

disp(['number of cross validation folds = ' num2str(nfolds)]);
disp(['number of arcing rounds = ' num2str(nrounds)]);
disp(['cross validation fold size = ' num2str(fsize)]);
disp(['classification threshold = ' num2str(thresh)]); 

tot_err = 0;
tot_spam_prec = 0;
tot_spam_rec = 0;
tot_legit_prec = 0;
tot_legit_rec = 0;
tot_weight_err = 0;
tot_weight_acc = 0;
tot_base_err = 0;
tot_base_acc = 0;
tot_ss = 0;
tot_sl = 0;
tot_ls = 0;
tot_ll = 0;
for k = 1:nfolds
    restinds = find(1:nfolds*fsize < 1+fsize*(k-1) | 1:nfolds*fsize > fsize*k);
    rest = data(:,restinds);
    restcharac = charac(restinds);
    restsize = length(restinds);
    on_probs_spam = zeros(nfeats-1,nrounds);
    off_probs_spam = zeros(nfeats-1,nrounds);
    on_probs_legit = zeros(nfeats-1,nrounds);
    off_probs_legit = zeros(nfeats-1,nrounds);
    spam_priors = zeros(1,nrounds);
    legit_priors = zeros(1,nrounds);
    misclass = zeros(1,restsize);
    p = ones(1,restsize)/restsize;
    for i = 1:nrounds
        s = sample(p,restsize);
        train = rest(:,s);
        traincharac = restcharac(s);
        spam = train(:,find(traincharac));
        legit = train(:,find(1-traincharac));
        num_spam = size(spam,2);
        num_legit = size(legit,2);
        temp = (sum(spam,2) + 1)/(num_spam + 2);
        on_prob_spam = log(temp);
        on_probs_spam(:,i) = on_prob_spam;
        off_prob_spam = log(1 - temp);
        off_probs_spam(:,i) = off_prob_spam;
        temp = (sum(legit,2) + 1)/(num_legit + 2);
        on_prob_legit = log(temp);
        on_probs_legit(:,i) = on_prob_legit;
        off_prob_legit = log(1 - temp);
        off_probs_legit(:,i) = off_prob_legit;
        spam_prior = log((num_spam + 1)/(restsize + 2));
        spam_priors(i) = spam_prior;
        legit_prior = log((num_legit + 1)/(restsize + 2));
        legit_priors(i) = legit_prior;
        for n = 1:restsize
            prob_spam = sum(on_prob_spam(rest(:,n) == 1)) + ...
                        sum(off_prob_spam(rest(:,n) == 0)) + spam_prior;
            prob_legit = sum(on_prob_legit(rest(:,n) == 1)) + ...
                         sum(off_prob_legit(rest(:,n) == 0)) + legit_prior;
            if prob_spam - prob_legit > log(thresh) & ~restcharac(n)
                misclass(n) = misclass(n) + 1;    
            elseif prob_spam - prob_legit <= log(thresh) & restcharac(n)
                misclass(n) = misclass(n) + 1;
            end
            p(n) = 1 + misclass(n)^4;
        end
        p = p/sum(p);
    end
    
    foldinds = 1+fsize*(k-1):fsize*k;
    spamsize = sum(charac(foldinds));
    legitsize = fsize - spamsize;
    fold = data(:,foldinds);
    ss = 0;
    sl = 0;
    ls = 0;
    ll = 0;
    num_errs = 0;
    if perfold
        disp([10 'fold ' num2str(k) ': #spam = ' num2str(spamsize) ...
             ', #legitimate = ' num2str(legitsize)]);
    end
    for n = 1:fsize
        votes = zeros(1,nrounds);
        for i = 1:nrounds
            on_prob_spam = on_probs_spam(:,i);
            off_prob_spam = off_probs_spam(:,i);
            on_prob_legit = on_probs_legit(:,i);
            off_prob_legit = off_probs_legit(:,i);
            spam_prior = spam_priors(i);
            legit_prior = legit_priors(i);
            prob_spam = sum(on_prob_spam(fold(:,n) == 1)) + ...
                        sum(off_prob_spam(fold(:,n) == 0)) + spam_prior;
            prob_legit = sum(on_prob_legit(fold(:,n) == 1)) + ...
                         sum(off_prob_legit(fold(:,n) == 0)) + legit_prior;
            if prob_spam - prob_legit > log(thresh) 
                votes(i) = 1;
            end
        end

        ind = fsize*(k-1)+n;
        if charac(ind) %spam
            if sum(votes) > (nrounds - 1)/2 
                ss = ss + 1;
            else
                sl = sl + 1;
                num_errs = num_errs + 1;
            end
        else %legitimate
            if sum(votes) > (nrounds - 1)/2
                ls = ls + 1;
                num_errs = num_errs + 1;
            else
                ll = ll + 1;
            end
        end
    end
    tot_ss = tot_ss + ss;
    tot_sl = tot_sl + sl;
    tot_ls = tot_ls + ls;
    tot_ll = tot_ll + ll;
    spam_prec = ss/(ss + ls);
    spam_rec = ss/(ss + sl);
    legit_prec = ll/(ll + sl);
    legit_rec = ll/(ll + ls);
    err = num_errs/fsize;
    weight_err = (lambda*ls + sl)/(lambda*legitsize + spamsize);
    weight_acc = (lambda*ll + ss)/(lambda*legitsize + spamsize);
    base_err = spamsize/(lambda*legitsize + spamsize); 
    base_acc = (lambda*legitsize)/(lambda*legitsize + spamsize);
    if perfold 
        disp(['spam recall = ' num2str(100*spam_rec) '%']);
        disp(['spam precision = ' num2str(100*spam_prec) '%']);
        disp(['weighted accuracy = ' num2str(100*weight_acc) '%']);
        disp(['baseline accuracy = ' num2str(100*base_acc) '%']);
        if allinfo 
            disp(['misclassification error = ' num2str(100*err) '%']);
            disp(['weighted error = ' num2str(100*weight_err) '%']);
            disp(['baseline error = ' num2str(100*base_err) '%']);
            disp(['legitimate precision = ' num2str(100*legit_prec) '%']);
            disp(['legitimate recall = ' num2str(100*legit_rec) '%']);
            disp('confusion matrix ([s->s, s->l; l->s, l->l]): ');
            disp(num2str([ss, sl; ls, ll],3));
        end
    end
    tot_err = tot_err + err;    
    tot_weight_err = tot_weight_err + weight_err;
    tot_weight_acc = tot_weight_acc + weight_acc;
    tot_base_err = tot_base_err + base_err;
    tot_base_acc = tot_base_acc + base_acc;
    tot_spam_prec = tot_spam_prec + spam_prec;
    tot_spam_rec = tot_spam_rec + spam_rec;
    tot_legit_prec = tot_legit_prec + legit_prec;
    tot_legit_rec = tot_legit_rec + legit_rec;
end    
tcr = tot_base_err/tot_weight_err;
disp([10 'statistics:' 10]);
disp(['legitimate message weight (lambda) = ' num2str(lambda)]);
disp(['number of features = ' num2str(nfeats-1)]);
disp(['mean spam recall = ' num2str(100*tot_spam_rec/nfolds) '%']);
disp(['mean spam precision = ' num2str(100*tot_spam_prec/nfolds) '%']);
disp(['mean weighted accuracy = ' num2str(100*tot_weight_acc/nfolds) '%']);
disp(['mean baseline accuracy = ' num2str(100*tot_base_acc/nfolds) '%']);
disp(['tcr = ' num2str(tcr)]);
if allinfo
    disp(['mean misclassification error = ' num2str(100*tot_err/nfolds) '%']);
    disp(['mean legitimate precision = ' ...
        num2str(100*tot_legit_prec/nfolds) '%']);
    disp(['mean legitimate recall = ' num2str(100*tot_legit_rec/nfolds) '%']);
    disp(['mean weighted error = ' num2str(100*tot_weight_err/nfolds) '%']);
    disp(['mean baseline error = ' num2str(100*tot_base_err/nfolds) '%']);
    disp('overall confusion matrix ([s->s, s->l; l->s, l->l]): ');
    disp(num2str([tot_ss, tot_sl; tot_ls, tot_ll],4));
    disp('mean confusion matrix ([s->s, s->l; l->s, l->l]/#cv_folds):  ');
    disp(num2str([tot_ss, tot_sl; tot_ls, tot_ll]/nfolds,4));
end
