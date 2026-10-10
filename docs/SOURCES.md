# المصادر الرسمية المعتمدة

اطّلعتُ عليها بتاريخ 2026‑10‑10 (نسخة جديدة جُلبت في هذه الجولة، لم تتغير مقاساتها عن الجولة الأولى) وبنيتُ عليها الموصل والفلاتر.
الوثائق تتغير؛ تحقق من النسخة الحالية قبل أي استخدام حقيقي. **لم أستطع** فتح صفحة `developers.binance.com/.../filters` و`.../CHANGELOG` بصيغتهما
المرسومة (HTML يعيد 202 فارغاً من هذه البيئة)، فاعتمدتُ نسخة المستودع الرسمي المكافئة (`filters.md`, `CHANGELOG.md`).

| المصدر | ما اعتُمد عليه |
|---|---|
| مستودع التوثيق الرسمي: <https://github.com/binance/binance-spot-api-docs> (`README.md`) | تأكيد اسم المنصة وواجهة `/api` وأن ما سواها «غير مدعوم» |
| `rest-api.md` — *General API Information* | عناوين الأساس؛ مضيف البيانات العامة `data-api.binance.vision`؛ مهلة 10 ثوانٍ؛ **5XX = حالة مجهولة**؛ HTTP 403/418/429؛ `Retry-After` |
| `rest-api.md` — *LIMITS* | وزن الطلبات `X-MBX-USED-WEIGHT-*`؛ حظر IP 2 دقيقة…3 أيام؛ حدود مبنية على IP |
| `rest-api.md` — *Request Security / SIGNED / Timing security* | HMAC‑SHA256؛ `timestamp`/`recvWindow` (≤ 60000، الموصى ≤ 5000)؛ قاعدة «الطابع ≤ خادم+1ث»؛ متجه اختبار التوقيع الرسمي |
| `rest-api.md` — *New order (TRADE)* | المعاملات؛ `newClientOrderId` «فريد بين المفتوحة» و**يُقبل ثانية بعد الاكتمال**؛ `newOrderRespType` |
| `rest-api.md` — *Test new order* | `POST /api/v3/order/test`: يتحقق من الصيغة والتوقيع والصلاحيات والفلاتر **ولا يصل إلى محرك المطابقة** |
| `rest-api.md` — *Cancel order*, *Query order*, *Current open orders*, *Account information*, *Account trade list* | استعلام/إلغاء بـ `origClientOrderId`؛ الأوزان (`myTrades` مع `orderId` = 5)؛ `omitZeroBalances`؛ حقل `uid` |
| `rest-api.md` — *Kline/Candlestick data*, *Exchange information* | شكل الشموع وحدّ 1000؛ شكل `exchangeInfo` والفلاتر |
| `filters.md` | PRICE_FILTER, LOT_SIZE, MARKET_LOT_SIZE, MIN_NOTIONAL, NOTIONAL, PERCENT_PRICE(_BY_SIDE), MAX_NUM_ORDERS, MAX_POSITION؛ والتقريب للأسفل |
| `enums.md` | حالات الأمر وأنواعه والجهة؛ `expiryReason`؛ حالة الرمز `CANCEL_ONLY` |
| `errors.md` | معاني ‎-1000/‎-1001/‎-1006/‎-1007/‎-1008 (حالة مجهولة)، ‎-1013، ‎-1021، ‎-2010/‎-2011/‎-2013 |
| `CHANGELOG.md` (حتى 2026‑09‑18) | الفلاتر `PERCENT_PRICE*`/`NOTIONAL` تستعمل «سعراً مرجعياً» حين يوجد؛ `expiryReason` في ردود الأمر؛ قاعدة النطاق السعري التنفيذية؛ `symbolStatus=CANCEL_ONLY`؛ تقاعد مخططات SBE القديمة (لا نستعمل SBE) |
| `faqs/api_key_types.md` | أنواع المفاتيح: Ed25519 (موصى به)، RSA، **HMAC (مهجور)** — الموصل يدعم HMAC فقط |
| <https://developers.binance.com/docs/wallet/account/api-key-permission> | `GET /sapi/v1/account/apiRestrictions` وحقوله (`enableWithdrawals`, `enableInternalTransfer`, `enableMargin`, `enableFutures`, `ipRestrict`…) |
| <https://www.binance.com/en/academy/articles/what-are-api-keys-and-security-types> (محدَّث 2026‑07‑06، قُرئ عبر أداة Firecrawl) | تخزين المفاتيح خارج Git، تدويرها، IP ثابت، الحساب الفرعي، **«Enable Spot & Margin Trading» تشمل الهامش**، «symbol whitelist»، حذف المفتاح فوراً عند الاشتباه، توصية Ed25519 بعبارة مرور |
| `testnet/general-info.md` (مستودع الوثائق) | Testnet: مفاتيح منفصلة، أرصدة وهمية تُصفَّر دورياً بلا إشعار، `/api` فقط **بلا `/sapi`** |
| `user-data-stream.md` | تدفق بيانات المستخدم الآن عبر WebSocket API (`userDataStream.subscribe`)؛ غير مستخدم هنا |

مكتبات قُيِّمت من PyPI (ترخيص/تاريخ آخر إصدار/اعتماديات): `httpx` (معتمدة)، `python-binance`، `binance-sdk-spot`،
`binance-connector` (مرفوضة لأسباب في [DECISIONS.md](DECISIONS.md))؛ وأدوات التطوير `pytest`, `hypothesis`, `ruff`, `mypy`, `pytest-cov`.
لم تُستنسخ أو تُشغَّل أي شيفرة من مستودعات خارجية، ولم تُثبَّت أي مهارة أو إضافة.
