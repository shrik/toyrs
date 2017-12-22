## pseudo code
data # user_id, film_id, score
matrix # film_ids X user_ids

predict(user, film) = neighbors(film).score_from(user)
neighbors()

precision
recall

Positive Negtive
      正样本 负样本
正确   60    20
错误   10    40

precision = 60/60+10
recall = 60 / 60 + 40

# TODO classification neighbors
# what if the rating, features is not number based

# How to describe the cross-validation process
