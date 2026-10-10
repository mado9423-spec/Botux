# المصادر الرسمية المعتمدة

اطّلعتُ عليها بتاريخ 2026‑10‑10 وبنيتُ عليها الموصل والفلاتر. (الوثائق تتغير؛ تحقق من النسخة الحالية قبل أي استخدام حقيقي.)

| المصدر | ما اعتُمد عليه |
|---|---|
| مستودع التوثيق الرسمي: <https://github.com/binance/binance-spot-api-docs> (`README.md`) | تأكيد اسم المنصة وواجهة `/api` وأن ما سواها «غير مدعوم» |
| `rest-api.md` — *General API Information* | عناوين الأساس؛ مضيف البيانات العامة `data-api.binance.vision`؛ مهلة 10 ثوانٍ؛ **5XX = حالة مجهولة**؛ HTTP 403/418/429؛ `Retry-After` |
| `rest-api.md` — *LIMITS* | وزن الطلبات `X-MBX-USED-WEIGHT-*`؛ حظر IP 2 دقيقة…3 أيام؛ حدود مبنية على IP |
| `rest-api.md` — *Request Security / SIGNED / Timing security* | HMAC‑SHA256؛ `timestamp`/`recvWindow` (≤ 60000، الموصى ≤ 5000)؛ قاعدة «الطابع ≤ خادم+1ث»؛ متجه اختبار التوقيع الرسمي |
| `rest-api.md` — *New order (TRADE)* | المعاملات؛ `newClientOrderId` «فريد بين المفتوحة» و**يُقبل ثانية بعد الاكتمال**؛ `newOrderRespType` |
| `rest-api.md` — *Test new order*, *Cancel order*, *Query order*, *Current open orders*, *Account information* | استعلام/إلغاء بـ `origClientOrderId`؛ الأوزان؛ `omitZeroBalances` |
| `rest-api.md` — *Kline/Candlestick data*, *Exchange information* | شكل الشموع وحدّ 1000؛ شكل `exchangeInfo` والفلاتر |
| `filters.md` | PRICE_FILTER, LOT_SIZE, MARKET_LOT_SIZE, MIN_NOTIONAL, NOTIONAL, PERCENT_PRICE(_BY_SIDE), MAX_NUM_ORDERS |
| `enums.md` | حالات الأمر وأنواعه والجهة |
| `errors.md` | معاني ‎-1000/‎-1001/‎-1006/‎-1007/‎-1008 (حالة مجهولة)، ‎-1013، ‎-1021، ‎-2010/‎-2011/‎-2013 |
| <https://developers.binance.com/docs/wallet/account/api-key-permission> | `GET /sapi/v1/account/apiRestrictions` وحقوله (`enableWithdrawals`, `enableInternalTransfer`, `enableMargin`, `enableFutures`, `ipRestrict`…) |

مكتبات قُيِّمت من PyPI (ترخيص/تاريخ آخر إصدار/اعتماديات): `httpx` (معتمدة)، `python-binance`، `binance-sdk-spot`،
`binance-connector` (مرفوضة لأسباب في [DECISIONS.md](DECISIONS.md))؛ وأدوات التطوير `pytest`, `hypothesis`, `ruff`, `mypy`, `pytest-cov`.
لم تُستنسخ أو تُشغَّل أي شيفرة من مستودعات خارجية.
