"""Internationalization (i18n) support for global deployment."""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class InternationalizationManager:
    """Manages internationalization and localization for the application."""
    
    def __init__(self, default_locale: str = "en_US"):
        """Initialize i18n manager.
        
        Args:
            default_locale: Default locale code (e.g., 'en_US', 'de_DE')
        """
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations: Dict[str, Dict[str, str]] = {}
        self.supported_locales = {
            "en_US": {"name": "English (US)", "rtl": False, "decimal_separator": ".", "date_format": "MM/DD/YYYY"},
            "en_GB": {"name": "English (UK)", "rtl": False, "decimal_separator": ".", "date_format": "DD/MM/YYYY"},
            "de_DE": {"name": "Deutsch (Deutschland)", "rtl": False, "decimal_separator": ",", "date_format": "DD.MM.YYYY"},
            "fr_FR": {"name": "Français (France)", "rtl": False, "decimal_separator": ",", "date_format": "DD/MM/YYYY"},
            "es_ES": {"name": "Español (España)", "rtl": False, "decimal_separator": ",", "date_format": "DD/MM/YYYY"},
            "ja_JP": {"name": "日本語 (日本)", "rtl": False, "decimal_separator": ".", "date_format": "YYYY/MM/DD"},
            "zh_CN": {"name": "中文 (中国)", "rtl": False, "decimal_separator": ".", "date_format": "YYYY/MM/DD"},
            "ar_SA": {"name": "العربية (السعودية)", "rtl": True, "decimal_separator": ".", "date_format": "DD/MM/YYYY"},
            "ko_KR": {"name": "한국어 (한국)", "rtl": False, "decimal_separator": ".", "date_format": "YYYY.MM.DD"},
            "pt_BR": {"name": "Português (Brasil)", "rtl": False, "decimal_separator": ",", "date_format": "DD/MM/YYYY"}
        }
        
        # Initialize default translations
        self._initialize_default_translations()
        
        logger.info(f"Initialized i18n manager with locale: {default_locale}")
    
    def _initialize_default_translations(self):
        """Initialize default translations for core functionality."""
        
        # English (US) - Base translations
        self.translations["en_US"] = {
            # Core application
            "app.name": "Counterfactual Lab",
            "app.description": "A data-generation studio for fairness and robustness research",
            
            # Generation
            "generation.started": "Counterfactual generation started",
            "generation.completed": "Generation completed successfully",
            "generation.failed": "Generation failed: {error}",
            "generation.samples": "{count} samples generated",
            "generation.attributes": "Attributes: {attributes}",
            
            # Evaluation
            "evaluation.bias_detected": "Bias detected in model outputs",
            "evaluation.fairness_score": "Fairness score: {score}",
            "evaluation.demographic_parity": "Demographic parity: {value}",
            "evaluation.no_bias": "No significant bias detected",
            
            # Errors and warnings
            "error.validation_failed": "Input validation failed",
            "error.model_load": "Failed to load model",
            "error.insufficient_memory": "Insufficient memory for operation",
            "warning.low_quality": "Generated samples may have low quality",
            "warning.bias_threshold": "Bias metrics exceed acceptable threshold",
            
            # System status
            "status.healthy": "System is healthy",
            "status.degraded": "System performance is degraded",
            "status.critical": "System is in critical state",
            "status.recovery": "System is recovering",
            
            # Compliance
            "compliance.audit_required": "Bias audit required",
            "compliance.gdpr_notice": "Processing personal data under GDPR",
            "compliance.data_retention": "Data will be retained for {days} days",
            "compliance.consent_required": "Explicit consent required for processing",
            
            # Reports
            "report.generated": "Report generated successfully",
            "report.bias_audit": "Bias Audit Report",
            "report.compliance": "Compliance Report",
            "report.performance": "Performance Report",
            
            # Common UI elements
            "ui.submit": "Submit",
            "ui.cancel": "Cancel",
            "ui.save": "Save",
            "ui.delete": "Delete",
            "ui.edit": "Edit",
            "ui.view": "View",
            "ui.download": "Download",
            "ui.upload": "Upload",
            "ui.loading": "Loading...",
            "ui.success": "Success",
            "ui.error": "Error",
            "ui.warning": "Warning",
            "ui.info": "Information"
        }
        
        # German (DE)
        self.translations["de_DE"] = {
            "app.name": "Kontrafaktisches Labor",
            "app.description": "Ein Datengenerierungs-Studio für Fairness- und Robustheitsforschung",
            
            "generation.started": "Kontrafaktische Generierung gestartet",
            "generation.completed": "Generierung erfolgreich abgeschlossen",
            "generation.failed": "Generierung fehlgeschlagen: {error}",
            "generation.samples": "{count} Proben generiert",
            "generation.attributes": "Attribute: {attributes}",
            
            "evaluation.bias_detected": "Bias in Modellausgaben erkannt",
            "evaluation.fairness_score": "Fairness-Score: {score}",
            "evaluation.demographic_parity": "Demografische Parität: {value}",
            "evaluation.no_bias": "Kein signifikanter Bias erkannt",
            
            "error.validation_failed": "Eingabevalidierung fehlgeschlagen",
            "error.model_load": "Modell konnte nicht geladen werden",
            "error.insufficient_memory": "Unzureichender Speicher für Operation",
            "warning.low_quality": "Generierte Proben könnten niedrige Qualität haben",
            "warning.bias_threshold": "Bias-Metriken überschreiten akzeptablen Schwellenwert",
            
            "status.healthy": "System ist gesund",
            "status.degraded": "Systemleistung ist verschlechtert",
            "status.critical": "System ist in kritischem Zustand",
            "status.recovery": "System erholt sich",
            
            "compliance.audit_required": "Bias-Audit erforderlich",
            "compliance.gdpr_notice": "Verarbeitung personenbezogener Daten unter DSGVO",
            "compliance.data_retention": "Daten werden {days} Tage gespeichert",
            "compliance.consent_required": "Ausdrückliche Einwilligung zur Verarbeitung erforderlich",
            
            "ui.submit": "Senden",
            "ui.cancel": "Abbrechen",
            "ui.save": "Speichern",
            "ui.delete": "Löschen",
            "ui.edit": "Bearbeiten",
            "ui.view": "Ansehen",
            "ui.download": "Herunterladen",
            "ui.upload": "Hochladen",
            "ui.loading": "Laden...",
            "ui.success": "Erfolg",
            "ui.error": "Fehler",
            "ui.warning": "Warnung",
            "ui.info": "Information"
        }
        
        # French (FR)
        self.translations["fr_FR"] = {
            "app.name": "Laboratoire Contrefactuel",
            "app.description": "Un studio de génération de données pour la recherche en équité et robustesse",
            
            "generation.started": "Génération contrefactuelle démarrée",
            "generation.completed": "Génération terminée avec succès",
            "generation.failed": "Échec de la génération : {error}",
            "generation.samples": "{count} échantillons générés",
            "generation.attributes": "Attributs : {attributes}",
            
            "evaluation.bias_detected": "Biais détecté dans les sorties du modèle",
            "evaluation.fairness_score": "Score d'équité : {score}",
            "evaluation.demographic_parity": "Parité démographique : {value}",
            "evaluation.no_bias": "Aucun biais significatif détecté",
            
            "compliance.audit_required": "Audit de biais requis",
            "compliance.gdpr_notice": "Traitement des données personnelles sous RGPD",
            "compliance.data_retention": "Les données seront conservées pendant {days} jours",
            "compliance.consent_required": "Consentement explicite requis pour le traitement",
            
            "ui.submit": "Soumettre",
            "ui.cancel": "Annuler",
            "ui.save": "Enregistrer",
            "ui.delete": "Supprimer",
            "ui.edit": "Modifier",
            "ui.view": "Voir",
            "ui.download": "Télécharger",
            "ui.upload": "Téléverser",
            "ui.loading": "Chargement...",
            "ui.success": "Succès",
            "ui.error": "Erreur",
            "ui.warning": "Avertissement",
            "ui.info": "Information"
        }
        
        # Spanish (ES)
        self.translations["es_ES"] = {
            "app.name": "Laboratorio Contrafactual",
            "app.description": "Un estudio de generación de datos para investigación en equidad y robustez",
            
            "generation.started": "Generación contrafactual iniciada",
            "generation.completed": "Generación completada exitosamente",
            "generation.failed": "Falló la generación: {error}",
            "generation.samples": "{count} muestras generadas",
            "generation.attributes": "Atributos: {attributes}",
            
            "evaluation.bias_detected": "Sesgo detectado en las salidas del modelo",
            "evaluation.fairness_score": "Puntuación de equidad: {score}",
            "evaluation.demographic_parity": "Paridad demográfica: {value}",
            "evaluation.no_bias": "No se detectó sesgo significativo",
            
            "compliance.audit_required": "Auditoría de sesgo requerida",
            "compliance.gdpr_notice": "Procesamiento de datos personales bajo RGPD",
            "compliance.data_retention": "Los datos se retendrán por {days} días",
            "compliance.consent_required": "Se requiere consentimiento explícito para el procesamiento",
            
            "ui.submit": "Enviar",
            "ui.cancel": "Cancelar",
            "ui.save": "Guardar",
            "ui.delete": "Eliminar",
            "ui.edit": "Editar",
            "ui.view": "Ver",
            "ui.download": "Descargar",
            "ui.upload": "Subir",
            "ui.loading": "Cargando...",
            "ui.success": "Éxito",
            "ui.error": "Error",
            "ui.warning": "Advertencia",
            "ui.info": "Información"
        }
        
        # Japanese (JP)
        self.translations["ja_JP"] = {
            "app.name": "反実仮想ラボ",
            "app.description": "公平性と堅牢性の研究のためのデータ生成スタジオ",
            
            "generation.started": "反実仮想生成が開始されました",
            "generation.completed": "生成が正常に完了しました",
            "generation.failed": "生成に失敗しました：{error}",
            "generation.samples": "{count}個のサンプルが生成されました",
            "generation.attributes": "属性：{attributes}",
            
            "evaluation.bias_detected": "モデル出力にバイアスが検出されました",
            "evaluation.fairness_score": "公平性スコア：{score}",
            "evaluation.demographic_parity": "人口統計学的パリティ：{value}",
            "evaluation.no_bias": "有意なバイアスは検出されませんでした",
            
            "compliance.audit_required": "バイアス監査が必要です",
            "compliance.gdpr_notice": "GDPRの下での個人データ処理",
            "compliance.data_retention": "データは{days}日間保持されます",
            "compliance.consent_required": "処理には明示的な同意が必要です",
            
            "ui.submit": "送信",
            "ui.cancel": "キャンセル",
            "ui.save": "保存",
            "ui.delete": "削除",
            "ui.edit": "編集",
            "ui.view": "表示",
            "ui.download": "ダウンロード",
            "ui.upload": "アップロード",
            "ui.loading": "読み込み中...",
            "ui.success": "成功",
            "ui.error": "エラー",
            "ui.warning": "警告",
            "ui.info": "情報"
        }
        
        # Chinese Simplified (CN)
        self.translations["zh_CN"] = {
            "app.name": "反事实实验室",
            "app.description": "用于公平性和鲁棒性研究的数据生成工作室",
            
            "generation.started": "反事实生成已开始",
            "generation.completed": "生成成功完成",
            "generation.failed": "生成失败：{error}",
            "generation.samples": "已生成{count}个样本",
            "generation.attributes": "属性：{attributes}",
            
            "evaluation.bias_detected": "在模型输出中检测到偏见",
            "evaluation.fairness_score": "公平性评分：{score}",
            "evaluation.demographic_parity": "人口统计学均等：{value}",
            "evaluation.no_bias": "未检测到显著偏见",
            
            "compliance.audit_required": "需要偏见审计",
            "compliance.gdpr_notice": "在GDPR下处理个人数据",
            "compliance.data_retention": "数据将保留{days}天",
            "compliance.consent_required": "处理需要明确同意",
            
            "ui.submit": "提交",
            "ui.cancel": "取消",
            "ui.save": "保存",
            "ui.delete": "删除",
            "ui.edit": "编辑",
            "ui.view": "查看",
            "ui.download": "下载",
            "ui.upload": "上传",
            "ui.loading": "加载中...",
            "ui.success": "成功",
            "ui.error": "错误",
            "ui.warning": "警告",
            "ui.info": "信息"
        }
    
    def set_locale(self, locale: str) -> bool:
        """Set the current locale.
        
        Args:
            locale: Locale code to set
            
        Returns:
            True if locale was set successfully
        """
        if locale in self.supported_locales:
            self.current_locale = locale
            logger.info(f"Locale set to: {locale}")
            return True
        else:
            logger.warning(f"Unsupported locale: {locale}")
            return False
    
    def get_supported_locales(self) -> Dict[str, Dict[str, Any]]:
        """Get list of supported locales with metadata."""
        return self.supported_locales.copy()
    
    def translate(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Translate a message key.
        
        Args:
            key: Translation key
            locale: Optional locale override
            **kwargs: Variables for string interpolation
            
        Returns:
            Translated string
        """
        target_locale = locale or self.current_locale
        
        # Try target locale first
        if target_locale in self.translations and key in self.translations[target_locale]:
            template = self.translations[target_locale][key]
        # Fall back to default locale
        elif self.default_locale in self.translations and key in self.translations[self.default_locale]:
            template = self.translations[self.default_locale][key]
        # Fall back to key itself
        else:
            template = key
            logger.debug(f"Translation not found for key: {key} in locale: {target_locale}")
        
        # Perform string interpolation
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning(f"String interpolation failed for key {key}: {e}")
            return template
    
    def add_translation(self, locale: str, key: str, value: str):
        """Add a single translation.
        
        Args:
            locale: Locale code
            key: Translation key
            value: Translated value
        """
        if locale not in self.translations:
            self.translations[locale] = {}
        
        self.translations[locale][key] = value
        logger.debug(f"Added translation {locale}.{key} = {value}")
    
    def load_translations_from_file(self, file_path: str, locale: str):
        """Load translations from a JSON file.
        
        Args:
            file_path: Path to translation file
            locale: Locale code for these translations
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            if locale not in self.translations:
                self.translations[locale] = {}
            
            self.translations[locale].update(translations)
            logger.info(f"Loaded translations for {locale} from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
    
    def export_translations(self, file_path: str, locale: Optional[str] = None):
        """Export translations to a JSON file.
        
        Args:
            file_path: Output file path
            locale: Optional specific locale to export (default: all)
        """
        try:
            if locale:
                data = {locale: self.translations.get(locale, {})}
            else:
                data = self.translations
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported translations to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export translations: {e}")
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format a number according to locale conventions.
        
        Args:
            number: Number to format
            locale: Optional locale override
            
        Returns:
            Formatted number string
        """
        target_locale = locale or self.current_locale
        locale_info = self.supported_locales.get(target_locale, self.supported_locales[self.default_locale])
        
        # Convert to string with appropriate decimal separator
        decimal_separator = locale_info.get("decimal_separator", ".")
        
        if decimal_separator == ",":
            # Use comma as decimal separator (European style)
            return f"{number:.3f}".replace(".", ",")
        else:
            # Use period as decimal separator (US/UK style)
            return f"{number:.3f}"
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format a date according to locale conventions.
        
        Args:
            date: Date to format
            locale: Optional locale override
            
        Returns:
            Formatted date string
        """
        target_locale = locale or self.current_locale
        locale_info = self.supported_locales.get(target_locale, self.supported_locales[self.default_locale])
        
        date_format = locale_info.get("date_format", "MM/DD/YYYY")
        
        # Simple format mapping
        format_map = {
            "MM/DD/YYYY": date.strftime("%m/%d/%Y"),
            "DD/MM/YYYY": date.strftime("%d/%m/%Y"),
            "DD.MM.YYYY": date.strftime("%d.%m.%Y"),
            "YYYY/MM/DD": date.strftime("%Y/%m/%d"),
            "YYYY.MM.DD": date.strftime("%Y.%m.%d")
        }
        
        return format_map.get(date_format, date.strftime("%Y-%m-%d"))
    
    def is_rtl_locale(self, locale: Optional[str] = None) -> bool:
        """Check if locale uses right-to-left text direction.
        
        Args:
            locale: Optional locale override
            
        Returns:
            True if locale is RTL
        """
        target_locale = locale or self.current_locale
        locale_info = self.supported_locales.get(target_locale, self.supported_locales[self.default_locale])
        return locale_info.get("rtl", False)
    
    def get_missing_translations(self, reference_locale: str = "en_US") -> Dict[str, List[str]]:
        """Find missing translations compared to reference locale.
        
        Args:
            reference_locale: Locale to use as reference
            
        Returns:
            Dictionary of missing translation keys per locale
        """
        if reference_locale not in self.translations:
            return {}
        
        reference_keys = set(self.translations[reference_locale].keys())
        missing = {}
        
        for locale in self.translations:
            if locale == reference_locale:
                continue
            
            locale_keys = set(self.translations[locale].keys())
            missing_keys = reference_keys - locale_keys
            
            if missing_keys:
                missing[locale] = list(missing_keys)
        
        return missing
    
    def validate_translations(self) -> Dict[str, List[str]]:
        """Validate all translations for issues.
        
        Returns:
            Dictionary of validation issues per locale
        """
        issues = {}
        
        for locale, translations in self.translations.items():
            locale_issues = []
            
            for key, value in translations.items():
                # Check for placeholder mismatches
                if "{" in value and "}" in value:
                    placeholders = re.findall(r'\{(\w+)\}', value)
                    # Could add more sophisticated validation here
                
                # Check for empty translations
                if not value.strip():
                    locale_issues.append(f"Empty translation for key: {key}")
                
                # Check for potentially missing translations (same as key)
                if value == key:
                    locale_issues.append(f"Potential missing translation: {key}")
            
            if locale_issues:
                issues[locale] = locale_issues
        
        return issues


# Global i18n manager instance
_global_i18n_manager = None

def get_global_i18n_manager(locale: str = "en_US") -> InternationalizationManager:
    """Get or create global i18n manager."""
    global _global_i18n_manager
    if _global_i18n_manager is None:
        _global_i18n_manager = InternationalizationManager(locale)
    return _global_i18n_manager

def initialize_i18n(locale: str = "en_US",
                   translation_files: Optional[Dict[str, str]] = None) -> InternationalizationManager:
    """Initialize global internationalization.
    
    Args:
        locale: Default locale
        translation_files: Optional mapping of locale -> file path
        
    Returns:
        Global i18n manager instance
    """
    global _global_i18n_manager
    _global_i18n_manager = InternationalizationManager(locale)
    
    # Load additional translation files if provided
    if translation_files:
        for locale_code, file_path in translation_files.items():
            if Path(file_path).exists():
                _global_i18n_manager.load_translations_from_file(file_path, locale_code)
    
    logger.info(f"Global i18n initialized with locale: {locale}")
    return _global_i18n_manager

def translate(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation.
    
    Args:
        key: Translation key
        locale: Optional locale override
        **kwargs: Variables for string interpolation
        
    Returns:
        Translated string
    """
    manager = get_global_i18n_manager()
    return manager.translate(key, locale, **kwargs)