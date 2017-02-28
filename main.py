from main_dt import decision_tree
from main_kmeans import kmeans
import time as tm

start = tm.clock()
decision_tree()
kmeans()
end = tm.clock()
print('Time taken:', round(end - start), 'sec')
