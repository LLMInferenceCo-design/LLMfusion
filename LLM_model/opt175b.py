from software_model.operators import Operator,Transpose,Concat, Reshape
from software_model.DataFrame import DataType,Tensor
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.communication_primitives import AllReduceMultiPCB
from software_model.gelu import GeLU
from software_model.mutmul_fusion import MatmulFusion
from software_model.matmul_horizontal_fusion import HorizontalMatmulFusion
from software_model.flash_attention_fusion import FlashAttentionFusion
from hardware_model.system import System

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
        self.ffn_fusion = []

        X = self.layer_norm0(X)


        # multi-head attention
        Q = self.Q_proj(X, self.Wq)
        assert Q.shape == [b, s, d // dev_cnt]
        Q = self.Q_reshape(Q, [b, s, h // dev_cnt, d_h])
        Q_T = self.Q_transpose(Q, [0, 2, 1, 3])
        assert Q_T.shape == [b, h // dev_cnt, s, d_h]

        K = self.K_proj(X, self.Wk)
        assert K.shape == [b, s, d // dev_cnt]
        K = self.K_reshape(K, [b, s, h // dev_cnt, d_h])
        K_T = self.K_transpose(K, [0, 2, 3, 1])
        assert K_T.shape == [b, h // dev_cnt, d_h, s]

        V = self.V_proj(X, self.Wv)
        assert V.shape == [b, s, d // dev_cnt]
        V = self.V_reshape(V, [b, s, h // dev_cnt, d_h])
        V_T = self.V_transpose(V, [0, 2, 1, 3])
        assert V_T.shape == [b, h // dev_cnt, s, d_h]

        A = self.Q_mul_K(Q_T, K_T)
        assert A.shape == [b, h // dev_cnt, s, s]
        A_softmax = self.A_softmax(A)

        H = self.A_mul_V(A_softmax, V_T)
        assert H.shape == [b, h // dev_cnt, s, d_h]
        H_T = self.H_transpose(H, [0, 2, 1, 3])
        assert H_T.shape == [b, s, h // dev_cnt, d_h]
        H = self.H_reshape(H_T, [b, s, d // dev_cnt])
        assert H.shape == [b, s, d // dev_cnt]

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

        # use start_time contral attention and ffn
        self.Q_fusion = MatmulFusion([self.Q_proj, self.Q_reshape, self.Q_transpose], self.data_type)
        self.K_fusion = MatmulFusion([self.K_proj, self.K_reshape, self.K_transpose],  self.data_type)
        self.V_fusion = MatmulFusion([self.V_proj, self.V_reshape, self.V_transpose], self.data_type)
        self.proj_fusion = HorizontalMatmulFusion([self.Q_fusion, self.K_fusion, self.V_fusion], self.data_type)

        self.flash_attention = FlashAttentionFusion([self.Q_mul_K, self.A_softmax, self.A_mul_V, self.H_transpose, self.H_reshape], self.data_type)

        self.H0_fusion = HorizontalMatmulFusion([MatmulFusion([self.H_matmul0], self.data_type)], self.data_type)

        # self.attention_fusion.append([self.Q_fusion, self.K_fusion, self.V_fusion])
        # self.attention_fusion.append([self.A_fusion])
        # self.attention_fusion.append([self.H_fusion])
        # self.attention_fusion.append([self.H0_fusion])



        self.H1_fusion = HorizontalMatmulFusion([MatmulFusion([self.H_matmul1, self.H_gelu], self.data_type)], self.data_type)
        self.H2_fusion = HorizontalMatmulFusion([MatmulFusion([self.H_matmul2], self.data_type)], self.data_type)
        # self.H2_fusion = MatmulFusion([self.H_matmul2], self.data_type)

        self.ffn_fusion.append([self.H1_fusion])
        self.ffn_fusion.append([self.H2_fusion])


    def compile_and_simulate(self, system: System):
        self.dag_construct()
        # compile and simulate the attention part
        if self.device_count>1:
            reduce_latency = self.allreduce_mha.simulate(system.interconnect) + self.allreduce_ffn.simulate(system.interconnect)
        else:
            reduce_latency = 0
        h1_latency = self.H1_fusion.compile_and_simulate(system.device)
        # print("h1_latency", h1_latency)
        layernorm_latency = self.layer_norm0.compile_and_simulate(system.device)
        proj_latency = self.proj_fusion.compile_and_simulate(system.device)
        attention_latency = self.flash_attention.compile_and_simulate(system.device)
        h0_latency = self.H0_fusion.compile_and_simulate(system.device)
        layernorm_latency += self.layer_norm1.compile_and_simulate(system.device)
        
        

       
        h2_latency = self.H2_fusion.compile_and_simulate(system.device)
        total_latency = layernorm_latency + proj_latency + attention_latency + h0_latency + h1_latency + h2_latency + reduce_latency
        return {
            "proj_latency": proj_latency,
            "attention_latency": attention_latency,
            "h0_latency": h0_latency,
            "h1_latency+gelu": h1_latency,
            "h2_latency": h2_latency,
            "layernorm_latency": layernorm_latency,
            "reduce_latency": reduce_latency,
            "total_latency": total_latency
        }



class opt175b_decode(Operator):

    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
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
        self.K_concat = Concat(data_type)
        self.V_concat = Concat(data_type)
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

    def __call__(self, x: Tensor, seq_len: int) -> Tensor:
        # b: batch size
        # s: sequence length
        # d: hidden dimension
        # d_h: dimension per head
        b, _, d = x.shape
        assert d == self.d_model
        s = seq_len
        h = self.n_heads
        dev_cnt = self.device_count
        d_h = d // h

        # KV cache
        K_cache = Tensor([b, h // dev_cnt, d_h, s], self.data_type)
        V_cache = Tensor([b, h // dev_cnt, s, d_h], self.data_type)

        # multi-head attention
        q = self.Q_proj(x, self.Wq)  # [b, 1, d / dev_cnt]
        assert q.shape == [b, 1, d // dev_cnt]
        k = self.K_proj(x, self.Wk)  # [b, 1, d / dev_cnt]
        v = self.V_proj(x, self.Wv)  # [b, 1, d / dev_cnt]
        q = self.Q_reshape(q, [b, 1, h // dev_cnt, d_h])
        k = self.K_reshape(k, [b, 1, h // dev_cnt, d_h])
        v = self.V_reshape(v, [b, 1, h // dev_cnt, d_h])
        q_T = self.Q_transpose(q, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert q_T.shape == [b, h // dev_cnt, 1, d_h]
        k_T = self.K_transpose(k, [0, 2, 3, 1])  # [b, h / dev_cnt, d_h, 1]
        assert k_T.shape == [b, h // dev_cnt, d_h, 1]
        v_T = self.V_transpose(v, [0, 2, 1, 3])  # [b, h / dev_cnt, 1, d_h]
        assert v_T.shape == [b, h // dev_cnt, 1, d_h]
        K_T = self.K_concat(K_cache, k_T, 3)  # [b, h / dev_cnt, d_h, s+1]
        assert K_T.shape == [b, h // dev_cnt, d_h, s + 1]
        V_T = self.V_concat(V_cache, v_T, 2)  # [b, h / dev_cnt, s+1, d_h]
        assert V_T.shape == [b, h // dev_cnt, s + 1, d_h]
        a = self.Q_mul_K(q_T, K_T)  # [b, h / dev_cnt, 1, s+1]
        assert a.shape == [b, h // dev_cnt, 1, s + 1]
        a_prob = self.A_softmax(a)
        h0 = self.A_mul_V(a_prob, V_T)  #  [b, h / dev_cnt, 1, d_h]
        assert h0.shape == [b, h // dev_cnt, 1, d_h]
        h0 = self.H_transpose(h0, [0, 2, 1, 3])  #  [b, 1, h / dev_cnt, d_h]
        assert h0.shape == [b, 1, h // dev_cnt, d_h]
        h0 = self.H_reshape(h0, [b, 1, d // dev_cnt])
        assert h0.shape == [b, 1, d // dev_cnt]
        h0 = self.H_matmul0(h0, self.W0)  #  [b, 1, d]
        assert h0.shape == [b, 1, d]
        h0 = self.layer_norm0(h0)
        assert h0.shape == [b, 1, d]
        if dev_cnt > 1:
            h0 = self.allreduce_mha(h0)

        # feed-forward network
        h1 = self.H_matmul1(h0, self.W1)  # [b, 1, 4 * d / dev_cnt]
        assert h1.shape == [b, 1, 4 * d // dev_cnt]
        h1 = self.H_gelu(h1)
        h2 = self.H_matmul2(h1, self.W2)  #  [b, 1, d]
        assert h2.shape == [b, 1, d]
        h2 = self.layer_norm1(h2)
        if dev_cnt > 1:
            h2 = self.allreduce_ffn(h2)

        assert h2.shape == [b, 1, d]
        self.memory_requirement = (
            self.Wq.size * self.Wq.data_type.word_size
            + self.Wk.size * self.Wk.data_type.word_size
            + self.Wv.size * self.Wv.data_type.word_size
            + self.W0.size * self.W0.data_type.word_size
            + self.W1.size * self.W1.data_type.word_size
            + self.W2.size * self.W2.data_type.word_size
            + K_cache.size * K_cache.data_type.word_size
            + V_cache.size * V_cache.data_type.word_size
        )
        return h2
    def dag_construct(self):

        # use start_time contral attention and ffn
        self.Q_fusion = MatmulFusion([self.Q_proj, self.Q_reshape, self.Q_transpose], self.data_type)
        self.K_fusion = MatmulFusion([self.K_proj, self.K_reshape, self.K_transpose],  self.data_type)
        self.V_fusion = MatmulFusion([self.V_proj, self.V_reshape, self.V_transpose], self.data_type)
        self.proj_fusion = HorizontalMatmulFusion([self.Q_fusion, self.K_fusion, self.V_fusion], self.data_type)
        self.Q_K_fusion = HorizontalMatmulFusion([MatmulFusion([self.Q_mul_K], self.data_type)], self.data_type)
        self.A_V_fusion = HorizontalMatmulFusion([MatmulFusion([self.A_mul_V], self.data_type)], self.data_type)

        self.H0_fusion = HorizontalMatmulFusion([MatmulFusion([self.H_matmul0], self.data_type)], self.data_type)

        # self.attention_fusion.append([self.Q_fusion, self.K_fusion, self.V_fusion])
        # self.attention_fusion.append([self.A_fusion])
        # self.attention_fusion.append([self.H_fusion])
        # self.attention_fusion.append([self.H0_fusion])



        self.H1_fusion = HorizontalMatmulFusion([MatmulFusion([self.H_matmul1, self.H_gelu], self.data_type)], self.data_type)
        self.H2_fusion = HorizontalMatmulFusion([MatmulFusion([self.H_matmul2], self.data_type)], self.data_type)
        # self.H2_fusion = MatmulFusion([self.H_matmul2], self.data_type)


    def compile_and_simulate(self, system: System):
        self.dag_construct()
        # compile and simulate the attention part
        q_mul_k_latency = self.Q_K_fusion.compile_and_simulate(system.device)
        layernorm_latency = self.layer_norm0.compile_and_simulate(system.device)
        proj_latency = self.proj_fusion.compile_and_simulate(system.device)
        
        softmax_latency = self.A_softmax.compile_and_simulate(system.device)
        a_mul_v_latency = self.A_V_fusion.compile_and_simulate(system.device)
        # attention_latency = self.flash_attention.compile_and_simulate(system.device)
        h0_latency = self.H0_fusion.compile_and_simulate(system.device)
        layernorm_latency += self.layer_norm1.compile_and_simulate(system.device)
        if self.device_count>1:
            reduce_latency = self.allreduce_mha.simulate(system.interconnect) + self.allreduce_ffn.simulate(system.interconnect)
        else:
            reduce_latency = 0

        h1_latency = self.H1_fusion.compile_and_simulate(system.device)
        h2_latency = self.H2_fusion.compile_and_simulate(system.device)
        total_latency = layernorm_latency + proj_latency + q_mul_k_latency + softmax_latency + a_mul_v_latency + h0_latency + h1_latency + h2_latency + reduce_latency
        return {
            
            "proj_latency": proj_latency,
            "q_mul_k_latency": q_mul_k_latency,
            "softmax_latency": softmax_latency,
            "a_mul_v_latency": a_mul_v_latency,
            "h0_latency": h0_latency,
            "h1_latency+gelu": h1_latency,
            "h2_latency": h2_latency,
            "layernorm_latency": layernorm_latency,
            "reduce_latency": reduce_latency,
            "total_latency": total_latency
        }