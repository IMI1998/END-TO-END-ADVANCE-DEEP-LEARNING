<div align="center">

# ⚡ Symplectic Padé-TabM: High-Throughput Tabular Deep Learning Engine
### آموزش ۵۰۰ میلیون سطر داده در ۳۵.۳۷ ثانیه روی ۲ کارت گرافیک Kaggle Tesla T4
**مبتنی بر PyTorch + OpenAI Triton + Lightning Fabric + Polars (بدون کدهای C/C++)**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenAI Triton](https://img.shields.io/badge/OpenAI_Triton-Fused_Kernels-00A67E?style=for-the-badge&logo=openai&logoColor=white)](https://triton-lang.org/)
[![Lightning Fabric](https://img.shields.io/badge/Lightning_Fabric-Multi--GPU-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![Hardware](https://img.shields.io/badge/Kaggle-2x_Tesla_T4_(16GB)-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.kaggle.com/)
[![Throughput](https://img.shields.io/badge/Throughput-14.13M_Rows%2Fs-blue?style=for-the-badge)]()
[![Validation R2](https://img.shields.io/badge/Out--of--Sample_R²-0.9998-brightgreen?style=for-the-badge)]()

</div>

---

## 📖 معرفی پروژه (Overview)

این مخزن پیاده‌سازی یک معماری یادگیری عمیق فوق‌سریع برای داده‌های جدولی (Tabular Deep Learning) در مقیاس صدها میلیون سطر است. این سیستم با ترکیب **تبدیل‌های ریاضی در فضای وزن (Weight-Space Isomorphism)**، **توابع فعال‌سازی کسری Padé-[2/2] در کرنل‌های یکپارچه Triton** و **آموزش توزیع‌شده ناهمگام با Lightning Fabric**، گلوگاه‌های پهنای باند حافظه (VRAM Bandwidth) و گذرگاه PCIe را حذف می‌کند.

در محیط استاندارد **Kaggle (مجهز به ۲ پردازنده گرافیکی NVIDIA Tesla T4)**، این معماری ۶ لایه با **۱۲۴,۳۸۹ پارامتر** و **۳۲ سرِ هم‌خانواده (32-Head Ensemble)** موفق شده است **۵۰۰,۰۰۰,۰۰۰ سطر (۵۰۰ میلیون نمونه)** را از حافظه اصلی سیستم (Cold Pinned RAM) استریم کرده و فرآیند کامل آموزش (Forward + Backward + Fused AdamW Step) را تنها در **۳۵.۳۷ ثانیه** (با نرخ **۱۴,۱۳۳,۰۲۰ سطر بر ثانیه**) به پایان برساند و به دقت برون‌نمونه‌ای **$R^2 = 0.9998$** دست یابد.

---

## 🧮 مبانی ریاضی و نوآوری‌های معماری (Mathematical Foundations)

### ۱. ایزومورفیسم رتبه-۱ در فضای وزن (Weight-Space Rank-1 TabM Isomorphism)
در معماری‌های استاندارد مبتنی بر آداپتورهای رتبه-۱ (Rank-1 Adapters)، اعمال بردارهای مقیاس‌دهی ورودی $\mathbf{s} \in \mathbb{R}^{D_{\text{in}}}$ و خروجی $\mathbf{r} \in \mathbb{R}^{D_{\text{out}}}$ در فضای فعال‌سازی (Activation Space) به صورت زیر انجام می‌شود:

$$\mathbf{Z} = \left( (\mathbf{H} \odot \mathbf{1}\mathbf{s}^\top) \mathbf{W} \right) \odot \mathbf{1}\mathbf{r}^\top$$

برای یک دسته (Batch) با ابعاد $B = 65,536$ و $D = 160$، این کار مستلزم دو بار خواندن و نوشتن تانسورهای عظیم فعال‌سازی در حافظه اصلی گرافیک (Global VRAM) است. ما با استفاده از خاصیت خطی ضرب تانسوری، این عملیات را به **فضای وزن** منتقل کرده‌ایم:

$$\mathbf{W}_{\text{eff}} = \mathbf{W} \odot (\mathbf{s} \otimes \mathbf{r}) = \mathbf{W} \odot (\mathbf{s}\mathbf{r}^\top), \qquad \mathbf{Z} = \mathbf{H} \mathbf{W}_{\text{eff}}$$

* **مزیت سخت‌افزاری:** ماتریس $\mathbf{W}_{\text{eff}} \in \mathbb{R}^{160 \times 160}$ تنها حاوی ۲۵,۶۰۰ درایه `FP16` (معادل **۵۰ کیلوبایت**) است که به طور کامل درون **کش سطح اول/دوم (L1/L2 Cache)** پردازنده گرافیک جای می‌گیرد و ترافیک حافظه اصلی را به صفر می‌رساند.
* **شاخه ورودی (Stem Dequantization Fusion):** در لایه اول، ضریب نرمال‌سازی داده‌های کوانتیزه‌شده `INT8` یعنی $\frac{1}{127}$ مستقیماً در بردار $\mathbf{s}_0$ ادغام می‌شود:

$$\mathbf{W}_{0,\text{eff}} = \mathbf{W}_0 \odot \left( \left(\frac{1}{127}\mathbf{s}_0\right) \mathbf{r}_0^\top \right)$$

---

### ۲. موجک کسری سیمپلکتیک Padé-[2/2] و مشتق کسر-حذف‌شده در Triton
به جای استفاده از توابع فعال‌سازی چندجمله‌ای یا استاندارد (مانند GELU/SiLU) که در تقریب توابع متناوب و غیرخطی پیچیده دچار افت می‌شوند، از تقریب کسری **Padé-[2/2]** به همراه پیوند باقی‌مانده حافظِ واریانس (Variance-Preserving Symplectic Residual) با ضریب $\beta = \frac{1}{\sqrt{2L}}$ استفاده شده است:

$$\phi(u) = \frac{u^2 + \alpha u}{1 + u^2}, \qquad \mathbf{H}^{(\ell+1)} = \mathbf{H}^{(\ell)} + \beta \, \phi\!\left(\mathbf{Z}^{(\ell)} + \mathbf{b}^{(\ell)}\right)$$

که در آن $\alpha \in \mathbb{R}$ یک پارامتر قابل یادگیری است که تقارن یا عدم تقارن موجک را تنظیم می‌کند.

#### بهینه‌سازی مشتق در کرنل بازگشت به عقب (`_pade_res_bwd_kernel`):
اگر از قاعده مشتق‌گیری خارج‌قسمت به شکل خام استفاده شود، عبارت $\frac{(2u+\alpha)(1+u^2) - (u^2+\alpha u)(2u)}{(1+u^2)^2}$ تولید می‌شود که نیازمند نگهداری صورت و مخرج در حافظه است. با ساده‌سازی جبری و حذف جملات متقارن $2u^3$، به فرمول بسته زیر می‌رسیم:

$$\frac{\partial \phi(u)}{\partial u} = \frac{2u + \alpha(1 - u^2)}{(1 + u^2)^2}, \qquad \frac{\partial \phi(u)}{\partial \alpha} = \frac{u}{1 + u^2}$$

* **مزیت سخت‌افزاری:** در کلاس `FusedSymplecticPadeFn`، کل عملیات جمع بایاس، تابع کسری، ضرب $\beta$ و جمع Residual در یک کرنل **OpenAI Triton** ادغام (Fuse) شده است. همچنین در مرحله Backward، تنها تانسور $\mathbf{Z}$ در `ctx.save_for_backward` ذخیره می‌شود و گرادیان $\alpha$ به صورت Atomic Reduction در ثبات‌های (Registers) کارت گرافیک محاسبه می‌گردد.

---

### ۳. کرنل یکپارچه تابع زیان ۳۲ سر و قضیه تنوع کروگ-ودلزبی (Krogh–Vedelsby Diversity)
در لایه خروجی، شبکه به صورت هم‌زمان $K = 32$ پیش‌بینی مستقل $\hat{y}_k$ را از طریق ماسک‌های متعامد تولید می‌کند. کرنل Triton اختصاصی (`_triton_tabm_mse_bwd`) خطای میانگین مربعات (MSE) و گرادیان‌های مقیاس‌دهی‌شده هر ۳۲ سر را در یک گذر واحد محاسبه می‌کند.

بر اساس **قضیه تجزیه کروگ-ودلزبی (Krogh–Vedelsby Decomposition)**، خطای کل مدل ترکیبی ($\hat{y}_{\text{ens}} = \frac{1}{K}\sum_{k=1}^K \hat{y}_k$) برابر است با:

$$\underbrace{\mathbb{E}\!\left[(\hat{y}_{\text{ens}} - y)^2\right]}_{\text{MSE}_{\text{ensemble}}} = \underbrace{\frac{1}{K}\sum_{k=1}^K \mathbb{E}\!\left[(\hat{y}_k - y)^2\right]}_{\bar{L} \text{ (میانگین خطای تک‌سرها)}} - \underbrace{\frac{1}{K}\sum_{k=1}^K \mathbb{E}\!\left[(\hat{y}_k - \hat{y}_{\text{ens}})^2\right]}_{D \text{ (تنوع هم‌خانواده / Diversity)}}$$

در این معماری، جمله تنوع $D$ به طور طبیعی خطای نهایی مدل را به اندازه **۵۰.۱٪** نسبت به میانگین خطای تک‌سرها ($\bar{L}$) کاهش می‌دهد.

---

### ۴. استریم داده با فشرده‌سازی `INT8` و آموزش دوگانه با Lightning Fabric
* **فشرده‌سازی ۴ برابری گذرگاه PCIe:** ویژگی‌های ورودی به صورت `int8` در حافظه قفل‌شده (Pinned Host Memory) به حجم ۱.۰۷ گیگابایت نگهداری می‌شوند و حجم انتقال داده از RAM به VRAM را به یک‌چهارمِ `float32` کاهش می‌دهند.
* **موازی‌سازی مستقل دو GPU با Local-SGD:** با استفاده از **Lightning Fabric** در مد `16-true` (دقت خالص `FP16`) و استریم‌های CUDA مجزا، هر کارت گرافیک Tesla T4 روی دسته‌های ۶۵,۵۳۶ سطری مجزا آموزش می‌بیند و هر ۲۰۰ گام، وزن‌های دو کارت گرافیک از طریق حافظه Peer-to-Peer با یکدیگر میانگین‌گیری می‌شوند.

---

## 📊 نتایج بنچمارک روی Kaggle (2x NVIDIA Tesla T4)

### مشخصات محیط تست و معماری
| پارامتر | مقدار / مشخصات |
| :--- | :--- |
| **سخت‌افزار** | Kaggle Dual GPU: `2x NVIDIA Tesla T4` (16 GB GDDR6 per GPU) |
| **پشته نرم‌افزاری** | PyTorch + OpenAI Triton + Lightning Fabric + Polars (Pure Python/Rust) |
| **تعداد کل سطرهای آموزش** | **۵۰۰,۰۰۰,۰۰۰ سطر** (۳,۸۱۴ گام × ۱۳۱,۰۷۲ سطر در هر گام سراسری) |
| **اندازه مخزن داده (Cold Pool)** | ۳۱,۴۵۷,۲۸۰ سطر (`1.07 GB` در Pinned Host Memory) |
| **مجموعه اعتبارسنجی (Unseen Val)** | ۱۳۱,۰۷۲ سطر کاملاً مجزا و دیده‌نشده در طول آموزش |
| **معماری شبکه** | ۶ لایه (`d_in=32`, `d_model=160`, `num_blocks=4`, `k_heads=32`) |
| **تعداد پارامترها** | ۱۱۹,۲۶۹ پارامتر قابل آموزش (۱۲۴,۳۸۹ پارامتر کل به همراه ماسک بافر) |

### جدول عملکرد و همگرایی
| شاخص ارزیابی (Metric) | نتیجه ثبت‌شده | توضیحات |
| :--- | :---: | :--- |
| **زمان کل آموزش (۵۰۰ میلیون سطر)** | **`35.37 s`** | شامل انتقال PCIe +‏ Forward +‏ Backward +‏ AdamW Step |
| **نرخ پردازش (Throughput)** | **`14,133,020 rows/s`** | بیش از ۱۴.۱۳ میلیون سطر در ثانیه |
| **حداکثر مصرف حافظه (Peak VRAM)** | **`732.4 MB`** | به ازای هر کارت گرافیک (کمتر از ۵٪ کل حافظه T4) |
| **خطای اولیه اعتبارسنجی (Initial MSE)** | `0.253927` | قبل از شروع آموزش |
| **خطای نهایی اعتبارسنجی (Final MSE)** | **`0.000039`** | کاهش خطا به میزان بیش از ۶,۵۰۰ برابر |
| **ضریب تعیین برون‌نمونه‌ای ($R^2$)** | **`0.9998`** | روی ۱۳۱,۰۷۲ نمونه دیده‌نشده (Unseen Validation Set) |
| **ممیزی تنوع کروگ-ودلزبی** | $\bar{L} = 0.000079, \; D = 0.000039$ | $\bar{L} - D = 0.000039$ (**۵۰.۱٪- کاهش MSE از طریق تنوع**) |

---

## 🖥️ خروجی کنسول (Raw Benchmark Log)

```text
[Stage 1] Building 1.07 GB Cold Pinned Pool (31.45M rows) & Unseen Validation Set...
[Stage 2] Streaming & Training ALL 500,000,000 rows (3,814 steps) with Formula-Boosted Fabric+Triton+PyTorch...
================================================================================================
BENCHMARK & MATHEMATICAL AUDIT (PYTORCH + TRITON + LIGHTNING FABRIC ON 2x TESLA T4)
================================================================================================
-> THROUGHPUT (500,000,000 ROWS)   :  14,133,020 rows/s | Total Time: 35.37 s
-> PEAK VRAM PER GPU               : 732.4 MB
-> UNSEEN VALIDATION CONVERGENCE   : MSE 0.253927 -> 0.000039 (Out-of-Sample R² = 0.9998)
-> KROGH-VEDELSBY DIVERSITY AUDIT  : L_bar (0.000079) - D (0.000039) = 0.000039 == 0.000039 (-50.1% MSE!)
================================================================================================
```

---

## 🚀 نحوه اجرا (Quick Start)

۱. در محیط **Kaggle Notebooks**، شتاب‌دهنده (Accelerator) را روی **GPU T4 x2** قرار دهید.
۲. کتابخانه‌های مورد نیاز به صورت پیش‌فرض نصب هستند (در صورت عدم وجود `lightning`، اسکریپت به طور خودکار آن را نصب می‌کند):
```bash
pip install torch triton lightning polars
```
۳. اسکریپت اصلی را اجرا کنید تا مخزن داده در حافظه Pinned ساخته شود و بنچمارک ۵۰۰ میلیون سطری به همراه ممیزی ریاضی اعتبارسنجی اجرا گردد.
