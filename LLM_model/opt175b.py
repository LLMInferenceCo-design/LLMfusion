from software_model.operators import Operator,Transpose,Concat, Reshape
from software_model.DataFrame import DataType,Tensor
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.gelu import GeLU
from software_model.layer_fusion import Operator_fusion

class opt175b_prefill(Operator):

    def __init__(self, d_model, n_head, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_head = n_head
        self.device_count = device_count
        # parameters per device
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        # operators per device
        # # multi-head attention
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.Q_reshape = Reshape(data_type)
        self.K_reshape = Reshape(data_type)
        self.V_reshape = Reshape(data_type)
        self.Q_transpose = Transpose(data_type)
        self.K_transpose = Transpose(data_type)
        self.V_transpose = Transpose(data_type)
        self.Q_mul_K = BatchedMatmul(data_type)
        self.A_softmax = Softmax(data_type)
        self.A_mul_V = BatchedMatmul(data_type)
        self.H_transpose = Transpose(data_type)
        self.H_reshape = Reshape(data_type)
        self.H_matmul0 = Matmul(data_type)
        self.layer_norm0 = LayerNorm(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        # # feed-forward network
        self.H_matmul1 = Matmul(data_type)
        self.H_gelu = GeLU(data_type)
        self.H_matmul2 = Matmul(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type)


    def __call__(self, X: Tensor):
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, s, d = X.shape
        assert d == self.d_model
        h = self.n_head
        d_h = d // h
        dev_cnt = self.device_count

        self.attention_fusion = []

        X = self.layer_norm0(X)


        # multi-head attention
        Q = self.Q_proj(X, self.Wq)
        assert Q.shape == [b, s, d // dev_cnt]
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]
        self.Q_fusion = Operator_fusion([self.Q_proj, self.Q_reshape, self.Q_transpose], [self.layer_norm0], self.data_type)

        K = self.K_proj(X, self.Wk)
        assert K.shape == [b, s, d // dev_cnt]
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        K_T = self.K_transpose(K, [0, 2, 1, 3])
        assert K_T.shape == [b, h // dev_cnt, s, d_h]
        self.K_fusion =Operator_fusion( [self.K_proj, self.K_reshape, self.K_transpose], [self.layer_norm0], self.data_type)

        V = self.V_proj(X, self.Wv)
        assert V.shape == [b, s, d // dev_cnt]
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
        V_T = self.V_transpose(V, [0, 2, 1, 3])
        assert V_T.shape == [b, h // dev_cnt, s, d_h]
        self.V_fusion =Operator_fusion( [self.V_proj, self.V_reshape,self.V_transpose], [self.layer_norm0], self.data_type)

        A = self.Q_mul_K(Q_T, K_T)
        assert A.shape == [b, h // dev_cnt, s, s]
        A_softmax = self.A_softmax(A)
        self.A_fusion = Operator_fusion([self.Q_mul_K, self.A_softmax], [self.Q_fusion, self.K_fusion], self.data_type)

        H = self.A_mul_V(A_softmax, V_T)
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H_T = self.H_transpose(H, [0, 2, 1, 3])
        assert H_T.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H_T, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]
        self.H_fusion = Operator_fusion([self.A_mul_V, self.H_transpose, self.H_reshape], [self.A_fusion], self.data_type)

        H0 = self.H_matmul0(H, self.W0)
        assert H0.shape == [b, s, d]
        if self.device_count > 1:
            H0 = self.allreduce_mha(H0)

        H0 = self.layer_norm1(H0)

        H1 = self.H_matmul1(H0, self.W1)
        assert H1.shape == [b, s, 4 * d // dev_cnt]
        H1 = self.H_gelu(H1)

        H2 = self.H_matmul2(H1, self.W2)
        assert H2.shape == [b, s, d]
        if self.device_count > 1:
            H2 = self.allreduce_ffn(H2)

        return H2

    def dag_construct(self):
        self.layers_list.append(self.layer_norm0)
        self.layers_list.append(self.Q_proj)
        self.layers_list.append(self.Q_reshape)
        self.layers_list.append(self.Q_transpose)
        self.layers_list.append(self.K_proj)
        self.layers_list.append(self.K_reshape)
        self.layers_list.append(self.K_transpose)
        self.layers_list.append(self.V_proj)
        self.layers_list.append(self.V_reshape)
        self.layers_list.append(self.V_transpose)
        self.layers_list.append(self.Q_mul_K)
        self.layers_list.append(self.A_softmax)
        self.layers_list.append(self.A_mul_V)
        self.layers_list.append(self.H_transpose)
        self.layers_list.append(self.H_reshape)
        self.layers_list.append(self.H_matmul0)
        self.layers_list.append(self.layer_norm1)
        self.layers_list.append(self.H_matmul1)
        self.layers_list.append(self.H_gelu)
        self.layers_list.append(self.H_matmul2)
        return self.layers_list






