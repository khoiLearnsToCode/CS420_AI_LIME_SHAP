#AI #XAI #SHAP #LRP #DeepLIFT #LIME 
# 1. Introduction
- Chúng tôi đưa ra góc nhìn xem bất kỳ lời giải thích nào cho dự đoán của một mô hình cũng là một mô hình, mà chúng tôi gọi là mô hình giải thích (explanation model). Điều này cho phép chúng tôi định nghĩa lớp các phương pháp quy gán đặc trưng dạng cộng (additive feature attribution methods) (Mục 2), lớp này thống nhất sáu phương pháp hiện có.
- Sau đó, chúng tôi chứng minh rằng các kết quả của lý thuyết trò chơi, vốn đảm bảo tồn tại một nghiệm duy nhất, áp dụng được cho toàn bộ lớp các phương pháp quy gán đặc trưng dạng cộng (Mục 3), và đề xuất giá trị SHAP như một thước đo thống nhất về mức độ quan trọng của đặc trưng mà các phương pháp khác nhau đều xấp xỉ (Mục 4).
- Chúng tôi đề xuất các phương pháp ước lượng giá trị SHAP mới, và chứng minh rằng chúng phù hợp với trực giác của con người hơn (được đo bằng các nghiên cứu người dùng) và phân biệt hiệu quả hơn giữa các lớp đầu ra của mô hình so với một số phương pháp hiện có (Mục 5).

## Note 
- x' được đề cập trong bài báo này có nghĩa là features vector representation của input x
- x' $\in$ $\{0,1\}^N$ ,  N là số feature. 0 tức là feature đó tắt, 1 là feature đó bật
# II. 
**2. Các phương pháp quy gán đặc trưng dạng cộng (Additive Feature Attribution Methods)**

Gọi $f$ là mô hình dự đoán gốc cần được giải thích và $g$ là mô hình giải thích, được thiết kế để giải thích một dự đoán $f(x)$ dựa trên một đầu vào duy nhất $x$, như đã đề xuất trong LIME [5]. Các mô hình giải thích thường dùng các _đầu vào đơn giản hóa_ $x'$, được ánh xạ về đầu vào gốc thông qua hàm ánh xạ $x = h_x(x')$. Các phương pháp cục bộ cố gắng đảm bảo $g(z') \approx f(h_x(z'))$ mỗi khi $z' \approx x'$. (Lưu ý rằng $h_x(x') = x$ mặc dù $x'$ có thể chứa ít thông tin hơn $x$, vì $h_x$ được xây dựng riêng cho đầu vào hiện tại $x$.)

**Định nghĩa 1.** _Các phương pháp quy gán đặc trưng dạng cộng có mô hình giải thích là một hàm tuyến tính của các biến nhị phân:_

$$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i, \qquad (1)$$

_trong đó $z' \in {0,1}^M$, $M$ là số đặc trưng đầu vào đơn giản hóa, và $\phi_i \in \mathbb{R}$._

Các phương pháp có mô hình giải thích thỏa Định nghĩa 1 sẽ gán một mức ảnh hưởng $\phi_i$ cho mỗi đặc trưng. Tổng các mức ảnh hưởng của tất cả đặc trưng sẽ xấp xỉ đầu ra $f(x)$ của mô hình gốc. Nhiều phương pháp hiện nay thỏa Định nghĩa 1, và một số trong đó được thảo luận dưới đây.

**Diễn giải công thức (1):** mỗi $z'_i$ cho biết đặc trưng thứ $i$ có mặt (1) hay vắng mặt (0). $\phi_0$ là giá trị nền, tức đầu ra khi không có đặc trưng nào. Mỗi $\phi_i$ là phần mà đặc trưng $i$ đóng góp vào dự đoán.

## 2.1 LIME
Để tìm $\phi$, LIME cực tiểu hóa hàm mục tiêu sau:

$$\xi = \underset{g \in \mathcal{G}}{\arg\min}\; L(f, g, \pi_{x'}) + \Omega(g). \qquad (2)$$

Mức độ trung thành (faithfulness) của mô hình giải thích $g(z')$ đối với mô hình gốc $f(h_x(z'))$ được đảm bảo thông qua hàm mất mát $L$, tính trên một tập mẫu trong không gian đầu vào đơn giản hóa và được đánh trọng số bởi nhân cục bộ (local kernel) $\pi_{x'}$. $\Omega$ là thành phần phạt độ phức tạp của $g$. Vì trong LIME, $g$ có dạng như Phương trình 1 và $L$ là hàm mất mát bình phương, nên Phương trình 2 có thể giải bằng hồi quy tuyến tính có phạt (penalized linear regression).

## 2.2 DeepLIFT

DeepLIFT sử dụng tính chất "tổng bằng độ chênh lệch" (summation-to-delta), được phát biểu như sau:

$$\sum_{i=1}^{n} C_{\Delta x_i \Delta o} = \Delta o, \qquad (3)$$

trong đó $o = f(x)$ là đầu ra của mô hình, $\Delta o = f(x) - f(r)$, $\Delta x_i = x_i - r_i$, và $r$ là đầu vào tham chiếu. Nếu đặt $\phi_i = C_{\Delta x_i \Delta o}$ và $\phi_0 = f(r)$, thì mô hình giải thích của DeepLIFT có dạng như Phương trình 1.

## 2.3 Lan truyền mức độ liên quan theo từng lớp (Layer-Wise Relevance Propagation)

Phương pháp _lan truyền mức độ liên quan theo từng lớp_ (LRP) dùng để diễn giải các dự đoán của mạng sâu [1]. Như Shrikumar và cộng sự đã chỉ ra, phương pháp này tương đương với DeepLIFT khi các giá trị kích hoạt tham chiếu của mọi nơ-ron đều được cố định bằng 0. Do đó, $x = h_x(x')$ chuyển các giá trị nhị phân về không gian đầu vào gốc: 1 nghĩa là đầu vào nhận giá trị gốc, còn 0 nghĩa là đầu vào nhận giá trị 0. Giống như DeepLIFT, mô hình giải thích của LRP cũng có dạng như Phương trình 1.

**Vì sao DeepLIFT khớp với Phương trình 1:** từ (3), ta có $f(x) = f(r) + \sum_i C_{\Delta x_i \Delta o}$. Như vậy dự đoán bằng giá trị nền $f(r)$ cộng với phần đóng góp của từng đặc trưng, đúng với dạng $g = \phi_0 + \sum_i \phi_i z'_i$ khi mọi $z'_i = 1$.

## 2.4 SHAP 

Phương pháp này đòi hỏi huấn luyện lại mô hình trên mọi tập con đặc trưng $S \subseteq F$, trong đó $F$ là tập tất cả các đặc trưng.
Nó gán cho mỗi đặc trưng một giá trị quan trọng, thể hiện ảnh hưởng của việc đưa đặc trưng đó vào mô hình lên dự đoán. 
Để tính ảnh hưởng này, ta huấn luyện một mô hình $f_{S\cup{i}}$ có chứa đặc trưng đó, và một mô hình $f_S$ không chứa nó. Sau đó, ta so sánh dự đoán của hai mô hình trên đầu vào hiện tại: $f_{S\cup{i}}(x_{S\cup{i}}) - f_S(x_S)$, trong đó $x_S$ là giá trị của các đặc trưng đầu vào thuộc tập $S$. Vì ảnh hưởng của việc loại bỏ một đặc trưng phụ thuộc vào các đặc trưng khác có trong mô hình, hiệu số trên được tính cho mọi tập con $S \subseteq F \setminus {i}$. Các giá trị Shapley sau đó được tính và dùng làm giá trị quy gán cho đặc trưng. Chúng là trung bình có trọng số của tất cả các hiệu số có thể:

$$\phi_i = \sum_{S \subseteq F\setminus{i}} \frac{|S|!,(|F|-|S|-1)!}{|F|!}\left[f_{S\cup{i}}(x_{S\cup{i}}) - f_S(x_S)\right]. \qquad (4)$$

Nếu đặt $\phi_0 = f_\varnothing(\varnothing)$, thì giá trị hồi quy Shapley có dạng như Phương trình 1, nên đây là một phương pháp quy gán đặc trưng dạng cộng.

_Giá trị lấy mẫu Shapley_ dùng để giải thích bất kỳ mô hình nào theo hai bước. (1) Áp dụng phép xấp xỉ bằng lấy mẫu cho Phương trình 4. (2) Xấp xỉ ảnh hưởng của việc bỏ một biến khỏi mô hình bằng cách lấy tích phân trên các mẫu từ tập dữ liệu huấn luyện. Nhờ đó không cần huấn luyện lại mô hình, và có thể tính ít hơn $2^{|F|}$ hiệu số. Vì mô hình giải thích của giá trị lấy mẫu Shapley có cùng dạng với giá trị hồi quy Shapley, nên nó cũng là một phương pháp quy gán đặc trưng dạng cộng.

# III. Các tính chất đặc biệt của Tính cộng tính 
## Tính chất 1: Độ chính xác cục bộ
$$f(x) = g(x') = \phi_0 + \sum_{i=1}^{M} \phi_i x'_i \qquad (5)$$

_Mô hình giải thích $g(x')$ khớp với mô hình gốc $f(x)$ khi $x = h_x(x')$, trong đó $\phi_0 = f(h_x(\mathbf{0}))$ biểu diễn đầu ra của mô hình khi tất cả các đầu vào đơn giản hóa đều bị tắt (tức là vắng mặt)._

**Diễn giải:** tại chính điểm cần giải thích, mọi đặc trưng đều có mặt, nên $x'_i = 1$ với mọi $i$. Khi đó Phương trình 5 trở thành:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

Nghĩa là **giá trị nền cộng với tổng đóng góp của mọi đặc trưng phải bằng đúng dự đoán**. Không được thừa, không được thiếu. 

## Tính chất 2: Tính vắng mặt
$$x'_i = 0 \implies \phi_i = 0 \qquad (6)$$

_Tính vắng mặt ràng buộc rằng các đặc trưng có $x'_i = 0$ không được gán bất kỳ mức ảnh hưởng nào._

## Tính chất 3: Tính nhất quán
 Tính nhất quán phát biểu rằng nếu mô hình thay đổi sao cho đóng góp của một đầu vào đơn giản hóa nào đó tăng lên hoặc giữ nguyên, bất kể các đầu vào khác ra sao, thì giá trị quy gán cho đầu vào đó không được giảm.

_Đặt $f_x(z') = f(h_x(z'))$, và ký hiệu $z' \setminus i$ nghĩa là đặt $z'_i = 0$. Với hai mô hình bất kỳ $f$ và $f'$, nếu_

$$f'_x(z') - f'_x(z' \setminus i) \geq f_x(z') - f_x(z' \setminus i) \qquad (7)$$

_với mọi đầu vào $z' \in {0,1}^M$, thì $\phi_i(f', x) \geq \phi_i(f, x)$._

**Định lý 1** _Chỉ có duy nhất một mô hình giải thích $g$ tuân theo Định nghĩa 1 và thỏa các Tính chất 1, 2 và 3:_

$$\phi_i(f, x) = \sum_{z' \subseteq x'} \frac{|z'|!,(M - |z'| - 1)!}{M!}\left[f_x(z') - f_x(z' \setminus i)\right] \qquad (8)$$

_trong đó $|z'|$ là số phần tử khác 0 của $z'$, và $z' \subseteq x'$ ký hiệu tất cả các vector $z'$ có tập phần tử khác 0 là tập con của tập phần tử khác 0 của $x'$._

Định lý 1 được suy ra từ việc kết hợp các kết quả của lý thuyết trò chơi hợp tác, trong đó các giá trị $\phi_i$ được gọi là giá trị Shapley [6]. Young (1985) đã chứng minh rằng giá trị Shapley là bộ giá trị duy nhất thỏa ba tiên đề tương tự Tính chất 1, Tính chất 3, và một tính chất cuối cùng mà chúng tôi chứng minh là thừa trong bối cảnh này (xem Tài liệu bổ sung). Tính chất 2 là cần thiết để điều chỉnh các chứng minh về giá trị Shapley sao cho áp dụng được cho lớp các phương pháp quy gán đặc trưng dạng cộng.

Dưới các Tính chất 1–3, với một phép ánh xạ đầu vào đơn giản hóa $h_x$ cho trước, Định lý 1 cho thấy chỉ tồn tại duy nhất một phương pháp quy gán đặc trưng dạng cộng. Kết quả này hàm ý rằng các phương pháp không dựa trên giá trị Shapley sẽ vi phạm độ chính xác cục bộ và/hoặc tính nhất quán (các phương pháp ở Mục 2 đều đã thỏa tính vắng mặt). Mục tiếp theo đề xuất một cách tiếp cận thống nhất nhằm cải tiến các phương pháp trước đây, giúp chúng không vô tình vi phạm Tính chất 1 và 3.

---

**Diễn giải nhanh ba tính chất:**

- **Tính vắng mặt:** đặc trưng vốn không có trong đầu vào thì không được nhận "công". Ví dụ với văn bản dạng túi từ, một từ không xuất hiện trong câu thì $\phi = 0$.
- **Tính nhất quán:** nếu ở mô hình $f'$, việc bật đặc trưng $i$ lên luôn tạo ra thay đổi lớn hơn hoặc bằng so với ở mô hình $f$, xét trong mọi bối cảnh $z'$, thì $\phi_i$ của $f'$ không được nhỏ hơn. Tính chất này ngăn trường hợp vô lý: mô hình phụ thuộc vào đặc trưng nhiều hơn mà điểm quan trọng của đặc trưng đó lại giảm.
- **Định lý 1:** Phương trình 8 chính là Phương trình 4 (giá trị Shapley) viết lại bằng ký hiệu nhị phân. $z'$ đóng vai trò của tập con $S$, $f_x(z')$ đóng vai trò của $v(S)$, và $z' \setminus i$ là "liên minh khi bỏ $i$". Điểm khác nằm ở chiều diễn đạt: Phương trình 4 tính trên các tập **không chứa** $i$ rồi thêm $i$ vào, còn Phương trình 8 tính trên các $z'$ rồi **bỏ** $i$ đi. Với các $z'$ không chứa $i$, hiệu số bằng 0, nên hai cách cho cùng kết quả.

Ý nghĩa thực tế: muốn có cả ba tính chất này thì **buộc phải dùng giá trị Shapley**. Các phương pháp như LIME (với kernel heuristic) hay DeepLIFT không cho ra giá trị Shapley, nên chúng vi phạm ít nhất một trong hai tính chất 1 và 3.


