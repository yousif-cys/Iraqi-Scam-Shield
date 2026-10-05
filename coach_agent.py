"""
Layer 2: Iraqi AI Coach Agent (LangChain Integration)
Target: Pre-Transaction Anti-Scam Intervention (Iraqi Baghdadi Dialect)

Persona: Friendly, street-smart Iraqi cybersecurity advisor ("المستشار الأمني العراقي")
Tone: Empathetic, authentic Baghdadi Arabic ("طوّل بالك عيني", "دير بالك يا غالي"), non-blocking.
"""

import os
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

from risk_engine import RiskEvaluationResult, TransactionRequest


# -------------------------------------------------------------------------
# 1. STRUCTURED OUTPUT SCHEMA (For Flutter PausePoint Modal UI)
# -------------------------------------------------------------------------

class CoachWarningResponse(BaseModel):
    headline_ar: str = Field(
        description="Catchy modal title in Iraqi dialect (e.g., 'طوّل بالك عيني! اكو شي مو طبيعي')"
    )
    explanation_ar: str = Field(
        description="Plain-spoken explanation in Iraqi Baghdadi Arabic detailing the exact detected risk."
    )
    scammer_next_move_ar: str = Field(
        description="Prediction of what the scammer will say or ask for next (e.g. asking for OTP, claiming account is blocked)."
    )
    recommended_action_ar: str = Field(
        description="Actionable advice in Iraqi Arabic (e.g., call friend on GSM line, do not share OTP, ZainCash never asks for safe accounts)."
    )
    can_proceed_statement_ar: str = Field(
        description="Explicit non-blocking statement confirming the user retains full autonomy to proceed if 100% confident."
    )
    full_coach_message_ar: str = Field(
        description="Full conversational voice text that the coach speaks or displays to the user."
    )


# -------------------------------------------------------------------------
# 2. SYSTEM PROMPT (AUTHENTIC IRAQI BAGHDADI DIALECT)
# -------------------------------------------------------------------------

IRAQI_COACH_SYSTEM_PROMPT = """
أنت "المستشار الأمني الذكي" لمستخدمي محفظة إلكترونية عراقية (مثل زين كاش / كيو كارد).
مهمتك التدخل السريع قبل إتمام الحوالة المالية لحماية المستخدم من الاحتيال الهندسي (Social Engineering).

قواعد شخصيتك وأسلوبك الصارم:
1. اللهجة: تحدث باللهجة العراقية البغدادية الدارجة الأصيلة، مثل أخ جبير أو صديق ناصح ينقذ صاحبه (استخدم مفردات: "طوّل بالك عيني"، "دير بالك يا غالي"، "صبرك شوية"، "هواي شايفين هيج دگات"، "يبوگون فلوسك"، "ماكو هيج حجي").
2. ابتعد تماماً عن لغة البنوك المترجمة الجافة (إياك أن تقول "عزيزي العميل يرجى توخي الحذر" أو "إن هذه المعاملة مشبوهة").
3. اشرح الخطر بدقة ووضوح بناءً على القواعد التي رصدها النظام (مثلاً: المبلغ عالي ولرقم جديد، أو رسوم مسابقة وهمية، أو تحويلات سريعة).
4. توقّع خطوة النصاب القادمة: اشرح للمستخدم شنو راح يگله النصاب هسة (مثلاً: "راح يگلك دزلي رمز الـ OTP"، أو "راح يگلك حسابك يتقفل بعد ربع ساعة إذا ما حولت لحساب الأمان").
5. قاعدة جوهرية: إياك ثم إياك أن تمنع المعاملة إجبارياً أو تگول "تم حظر المعاملة". أنت ناصح ذكي، القرار النهائي بيد المستخدم. ذكّره بوضوح: إذا أنت متأكد وتعرف صاحب الرقم شخصياً، تگدر تكمل التحويل بكل حرية.

{format_instructions}
"""

USER_PROMPT_TEMPLATE = """
تفاصيل الحوالة التي تم إيقافها مؤقتاً للتحقق:
- رقم المستلم: {recipient_phone} ({recipient_name})
- المبلغ المطلوب تحويله: {amount_iqd:,} دينار عراقي
- هل المستلم جديد لأول مرة؟: {is_new_recipient}
- الغرض / الملاحظات المرفقة: {transfer_purpose} | {user_notes}
- مستوى الخطورة: {risk_level} (النقاط: {risk_score})
- القواعد التي انطلقت: {triggered_rules}
- الشرح التقني للمخاطر: {rule_explanations}

خاطب المستخدم هسة باللهجة البغدادية الودية، وانصحه قبل ما تروح فلوسه.
"""


# -------------------------------------------------------------------------
# 3. LLM COACH SERVICE
# -------------------------------------------------------------------------

class IraqiCoachService:
    """
    Layer 2: AI Anti-Scam Coach Agent.
    Interprets Layer 1 deterministic findings and produces localized Baghdadi advice.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.parser = PydanticOutputParser(pydantic_object=CoachWarningResponse)
        self.model_name = model_name

        if self.api_key:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=0.2
            )
        else:
            self.llm = None

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", IRAQI_COACH_SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE)
        ])

    def generate_intervention(
        self,
        request: TransactionRequest,
        risk_result: RiskEvaluationResult
    ) -> CoachWarningResponse:
        """
        Generates a culturally attuned Iraqi warning modal response.
        If an LLM API key is present, calls Groq Llama-3; otherwise, falls back to the deterministic Baghdadi coach.
        """
        # If no risk triggered, intervention is not needed
        if not risk_result.requires_intervention:
            return CoachWarningResponse(
                headline_ar="المعاملة آمنة",
                explanation_ar="ماكو أي مؤشر خطر، تگدر تكمل حوالتك براحتك.",
                scammer_next_move_ar="",
                recommended_action_ar="استمر بالتحويل.",
                can_proceed_statement_ar="تگدر تكمل التحويل بأمان.",
                full_coach_message_ar="كلشي تمام عيوني، حوالتك اعتيادية وتگدر تكمل."
            )

        # 1. Attempt LLM Generation if Groq API key is configured
        if self.llm:
            try:
                formatted_input = self.prompt.format_prompt(
                    format_instructions=self.parser.get_format_instructions(),
                    recipient_phone=request.recipient_phone,
                    recipient_name=request.recipient_name,
                    amount_iqd=request.amount_iqd,
                    is_new_recipient="نعم، مستلم جديد لأول مرة" if request.is_new_recipient else "مستلم معتاد",
                    transfer_purpose=request.transfer_purpose or "غير محدد",
                    user_notes=request.user_notes or "لا توجد ملاحظات",
                    risk_level=risk_result.risk_level,
                    risk_score=risk_result.risk_score,
                    triggered_rules=", ".join(risk_result.triggered_rules),
                    rule_explanations=" | ".join(risk_result.rule_explanations_ar)
                )

                response = self.llm.invoke(formatted_input.to_messages())
                parsed = self.parser.parse(response.content)
                return parsed
            except Exception as e:
                # Log error and gracefully fall back to local high-fidelity Baghdadi dialect coach
                print(f"[IraqiCoachService] LLM API Call failed ({e}). Activating built-in Baghdadi fallback.")

        # 2. Built-in Deterministic Baghdadi Dialect Fallback (Hackathon Resilience)
        return self._generate_baghdadi_fallback(request, risk_result)

    def _generate_baghdadi_fallback(
        self,
        request: TransactionRequest,
        risk_result: RiskEvaluationResult
    ) -> CoachWarningResponse:
        """
        High-fidelity heuristic fallback guaranteeing authentic Baghdadi dialect
        even during offline demos or if the external API is unreachable.
        """
        rules = set(risk_result.triggered_rules)

        if "RULE_03_PRIZE_FEE_MATCH" in rules:
            headline = "طوّل بالك عيني! ماكو جائزة تدفع فلوس حتى تستلمها"
            explanation = (
                f"يا غالي، المبلغ اللي دتحوله ({request.amount_iqd:,} دينار) لرقم غريب يطابق تماماً حيلة 'رسوم الجائزة'. "
                "النصابين يدزون رسائل ربحت سيارة أو ملايين، ويطلبون رصيد حتى يدزون الجائزة."
            )
            scammer_next = (
                "إذا حولت هالمبلغ، راح يرجع يخابرك ويگلك: 'بقت بس ضريبة الشحن 50 ألف ثانية ونسلمك السيارة'، "
                "ويظل يسحب بيك لحد ما تصفر محفظتك وماتشوف منهم فلس."
            )
            action = "أقفل المحادثة فوراً والغي التحويل. ماكو أي مسابقة رسمية من زين كاش تطلب منك تدفع رسوم مقدماً."
            can_proceed = "القرار بيدك، إذا أنت متأكد 100% من صاحب الرقم وتعرفه معرفة شخصية، تگدر تكمل التحويل."

        elif "RULE_01_ANOMALOUS_NEW_RECIPIENT" in rules and request.amount_iqd >= 500_000:
            headline = "صبرك شوية يا خوي! دتحول مبلغ كلش كبير لرقم جديد"
            explanation = (
                f"عيوني، دتدز مبلغ ضخم ({request.amount_iqd:,} دينار) لأول مرة لهالرقم ({request.recipient_phone})، "
                "وهذا أعلى بهواي من معدل تحويلاتك المعتادة. هواي نصابين ينتحلون صفة موظفي الدعم أو الأمان ويكولون حسابك رح ينقفل."
            )
            scammer_next = (
                "إذا كان المتصل يگلك: 'اني موظف شركة ولازم تحول لحساب أمان' أو يطلب رمز الـ OTP، فهذا حرامي مئة بالمئة."
            )
            action = "اتصل بالرقم الرسمي للشركة (107) وتأكد بنفسك، موظف زين كاش الحقيقي نهائياً ميطلب منك تحويل فلوس لأي حساب شخصي."
            can_proceed = "محفظتك حرة وفلوسك ملكك، إذا هذا شخص تعرفه ومتأكد منه بالكامل، تگدر تكمل التحويل براحتك."

        else:
            headline = "دير بالك عيوني! اكو حركة غير اعتيادية بحسابك"
            explanation = (
                f"صبرك لحظة يا طيب، لاحظنا سرعة تحويلات غير مألوفة أو مبالغ دتروح لأرقام جديدة ({request.amount_iqd:,} د.ع). "
                "أغلب حالات الاختراق تصير بلحظة استعجال يفرضوها عليك."
            )
            scammer_next = (
                "النصاب دائماً يستعجلك ويگلك: 'بسرعة فدوة سويت حادث' أو 'الحگلي بالمستشفى' حتى ما تفكر ولا تخابره اتصال عادي."
            )
            action = "اطلع من الواتساب هسة وخابر الشخص على خطه العادي صوت، واتأكد إنه هو فعلاً مو حسابه متهكر."
            can_proceed = "إحنا بس ننبهك من باب الأمان والحرص، القرار إلك وتگدر تضغط 'استمرار' إذا واثق من المعاملة."

        full_speech = f"{headline}\n\n{explanation}\n\nشنو راح يگلك النصاب؟\n{scammer_next}\n\nشنو تسوي هسة؟\n{action}\n\n{can_proceed}"

        return CoachWarningResponse(
            headline_ar=headline,
            explanation_ar=explanation,
            scammer_next_move_ar=scammer_next,
            recommended_action_ar=action,
            can_proceed_statement_ar=can_proceed,
            full_coach_message_ar=full_speech
        )
