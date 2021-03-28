class Polynomial:

    def __init__(self, *args):
        self.coefs = args

    def __call__(self, x):
        ans = 0
        extent_x = 1
        for coef in self.coefs:
            ans += coef * extent_x
            extent_x *= x
        return ans
