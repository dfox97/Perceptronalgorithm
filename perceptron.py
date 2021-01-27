import nump as np

class  Perceptron:
    def __init__(self,l_rate=0.01,n_iters=1000):
        self.lr=l_rate
        self.n_iters=n_iters
        self.activation_func=self._unit_step_func
        self.weights=None
        self.bias =None

    #training samples and labels
    def fit(self,x,y):
        n_samples, n_features=x.shape

        self.weights=np.zeros(n_features)
        self.bias=0
        y_=np.array([1 if i>0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(x):
                linear_output=np.dot(x_i,self.weights)+self.bias
                y_predicted=self.activation_func(linear_output)

                update=self.lr * (y_[idx]-y_predicted)
                self.weights+=update*x_i
                self.bias+=update
                object
    def predict(self,x):
        linear_output=np.dot(x,self.weights)+self.bias
        y_predicted=self.activation_func(linear_output)
        return y_predicted


    def _unit_step_func(self,x):
        return np.where(x>=0,1,0)
