Shape (21613, 21)

No missing values 

All unique



- feature "id" doesnt give any signal
- feature date probably doesn't give any signal


- sqft_living15 sometimes have different value from sqft_living. However the yr_renovated at the same time sometimes is 0

-----

- *price* feature is very right skewed

- bedroom -- normal, categorical?

- bathroom -- weird 

- *sqft_living* -- "normal" a bit right skewed

- *sqft_lot* -- WTF is this distribution??? UPD: np.log kinda normilizes it, but it is still not normal

- floors - a bit weird. Categorical?? What the hell is 1.5 floors

- waterfront - categorical, unbalanced. VERY UNBALANCED

- view - categorical

- condition - 

- grade - kinda normal

- *sqft_above* -- "normal" right skewed

- *sqft_basement* -- high spike where there is no basement

- yr_build -- almost uniform

- yr_renovated -- high spike on 0 where there is no renovation

- zipcode - categorical, 

- lat 

- long 

- *sqft_living15* - "normal" very close to normal, right skewed. No 0 values. Frequently different value from the sqft_living, which means renovations were made, but there is no record of such TODO

sqft_lot15 -- high spike near 0


What are 0 bathrooms and 0 bedrooms lots. Can they be the empty lots without a house for sale? The same time min floor is 1... 


Let's check the Z-Score for all the other features for the potential outliers 

==================================================
ID: 9714
Price z-score: 1.0825159676954192
sqft_living z-score: 3.7728943038821985
sqft_lot z-score: 20.66881375565044
sqft_lot15 z-score: 31.44029287005133

==================================================
ID: 20452
Price z-score: 2.887109398264106
sqft_living z-score: 4.845388779214749
sqft_lot z-score: 20.66881375565044
sqft_lot15 z-score: 30.961673753674678

==================================================
ID: 13464
Price z-score: 0.6807385623989947
sqft_living z-score: 0.6098522218100552
sqft_lot z-score: 10.065995393284764
sqft_lot15 z-score: 20.065104545777597

Looks like it is an outlier, not an input error
- The Price z-score < 2 -- Not outlier for the price 
- The sqft_living > 3 -- Outlier
- The sqft_lot > 20 -- WTF! Outlier
- The sqft_living15 > 31 -- DEAD Outlier


Conclusions:
- We need to log_transform 
    - For EDA: transformation helps you see relationships more clearly — do it selectively for visualization
For modeling: transform skewed features to meet linear regression assumptions, but tree-based models (Random Forest, XGBoost) don't require it at all  
- We can replace the zip with the districts for binning