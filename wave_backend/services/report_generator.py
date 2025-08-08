"""
Report Generator
Generate professional trading reports with comprehensive analytics.
"""

import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import base64
from io import BytesIO

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, Color
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

# Chart generation
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

from ..config.settings import get_settings
from ..services.event_bus import EventBus

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Individual report section data."""
    title: str
    content_type: str  # 'text', 'table', 'chart', 'metrics'
    data: Any
    description: Optional[str] = None
    chart_config: Optional[Dict] = None

@dataclass
class ReportMetrics:
    """Core trading metrics for reports."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_streak: int
    max_winning_streak: int
    max_losing_streak: int

@dataclass
class RiskMetrics:
    """Risk analysis metrics."""
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    conditional_var_99: float
    beta: float
    correlation_to_market: float
    tracking_error: float
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    
@dataclass
class AttributionAnalysis:
    """Performance attribution breakdown."""
    strategy_attribution: Dict[str, float]
    asset_attribution: Dict[str, float]
    sector_attribution: Dict[str, float]
    time_attribution: Dict[str, float]
    factor_attribution: Dict[str, float]

@dataclass
class ComplianceMetrics:
    """Regulatory compliance metrics."""
    position_limits_breached: int
    concentration_limits_breached: int
    exposure_limits_breached: int
    trading_hours_violations: int
    order_size_violations: int
    wash_sale_violations: int
    pattern_day_trading_violations: int
    regulatory_alerts: List[Dict]

@dataclass
class GeneratedReport:
    """Complete generated report."""
    report_id: str
    report_type: str
    generated_at: datetime
    period_start: date
    period_end: date
    
    # Report content
    executive_summary: Dict
    performance_metrics: ReportMetrics
    risk_analysis: RiskMetrics
    attribution_analysis: AttributionAnalysis
    compliance_metrics: ComplianceMetrics
    sections: List[ReportSection]
    
    # Export information
    pdf_path: Optional[str] = None
    json_export: Optional[str] = None
    generation_time: float = 0.0
    file_size_bytes: int = 0

class ReportGenerator:
    """Generate professional trading reports."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.settings = get_settings()
        
        # Report settings
        self.report_dir = Path("reports")
        self.report_dir.mkdir(exist_ok=True)
        
        # Chart settings
        self.chart_style = 'seaborn-v0_8'
        plt.style.use('default')
        sns.set_palette("husl")
        
        # PDF settings
        self.page_size = A4
        self.margin = 0.75 * inch
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#1f4788')
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=HexColor('#2d5aa0')
        )
        
        # Template configurations
        self.report_templates = {
            'daily': self._get_daily_template(),
            'monthly': self._get_monthly_template(),
            'quarterly': self._get_quarterly_template(),
            'annual': self._get_annual_template(),
            'custom': self._get_custom_template()
        }
        
        # Color scheme
        self.colors = {
            'primary': HexColor('#1f4788'),
            'secondary': HexColor('#2d5aa0'),
            'success': HexColor('#28a745'),
            'danger': HexColor('#dc3545'),
            'warning': HexColor('#ffc107'),
            'info': HexColor('#17a2b8')
        }
        
    async def daily_pnl_report(self, report_date: Union[str, date]) -> GeneratedReport:
        """Generate detailed daily P&L breakdown."""
        
        start_time = datetime.utcnow()
        
        if isinstance(report_date, str):
            report_date = datetime.strptime(report_date, '%Y-%m-%d').date()
        
        logger.info(f"Generating daily P&L report for {report_date}")
        
        # Get daily trading data
        daily_data = await self._fetch_daily_data(report_date)
        
        # Calculate daily metrics
        daily_metrics = await self._calculate_daily_metrics(daily_data)
        
        # Generate sections
        sections = []
        
        # Executive summary
        exec_summary = await self._create_daily_executive_summary(daily_data, daily_metrics)
        sections.append(ReportSection(
            title="Executive Summary",
            content_type="text",
            data=exec_summary,
            description="High-level overview of daily performance"
        ))
        
        # P&L breakdown
        pnl_table = await self._create_pnl_breakdown_table(daily_data)
        sections.append(ReportSection(
            title="P&L Breakdown",
            content_type="table",
            data=pnl_table,
            description="Detailed profit and loss analysis by strategy and position"
        ))
        
        # Performance chart
        perf_chart = await self._create_daily_performance_chart(daily_data)
        sections.append(ReportSection(
            title="Daily Performance",
            content_type="chart",
            data=perf_chart,
            description="Intraday performance visualization"
        ))
        
        # Top performers/detractors
        performers_data = await self._analyze_daily_performers(daily_data)
        sections.append(ReportSection(
            title="Top Performers & Detractors",
            content_type="table",
            data=performers_data,
            description="Best and worst performing positions of the day"
        ))
        
        # Risk metrics
        risk_metrics = await self._calculate_daily_risk_metrics(daily_data)
        
        # Compliance check
        compliance_metrics = await self._check_daily_compliance(daily_data)
        
        # Attribution analysis
        attribution = await self._perform_daily_attribution(daily_data)
        
        # Create report
        report = GeneratedReport(
            report_id=f"daily_{report_date.strftime('%Y%m%d')}",
            report_type="daily_pnl",
            generated_at=datetime.utcnow(),
            period_start=report_date,
            period_end=report_date,
            executive_summary=exec_summary,
            performance_metrics=daily_metrics,
            risk_analysis=risk_metrics,
            attribution_analysis=attribution,
            compliance_metrics=compliance_metrics,
            sections=sections,
            generation_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
        # Export to PDF
        pdf_path = await self._export_report_to_pdf(report)
        report.pdf_path = str(pdf_path)
        report.file_size_bytes = pdf_path.stat().st_size
        
        logger.info(f"Daily P&L report generated: {pdf_path}")
        
        return report
    
    async def monthly_performance_report(self, month: str) -> GeneratedReport:
        """Generate monthly performance report with comprehensive analytics."""
        
        start_time = datetime.utcnow()
        
        # Parse month (YYYY-MM format)
        year, month_num = map(int, month.split('-'))
        period_start = date(year, month_num, 1)
        
        # Calculate end date
        if month_num == 12:
            period_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            period_end = date(year, month_num + 1, 1) - timedelta(days=1)
        
        logger.info(f"Generating monthly performance report for {month}")
        
        # Fetch monthly data
        monthly_data = await self._fetch_period_data(period_start, period_end)
        
        # Calculate comprehensive metrics
        performance_metrics = await self._calculate_period_metrics(monthly_data)
        risk_metrics = await self._calculate_period_risk_metrics(monthly_data)
        attribution = await self._perform_period_attribution(monthly_data)
        compliance_metrics = await self._check_period_compliance(monthly_data)
        
        # Generate report sections
        sections = []
        
        # Executive summary
        exec_summary = await self._create_monthly_executive_summary(
            monthly_data, performance_metrics
        )
        
        # Performance analysis
        sections.extend([
            ReportSection(
                title="Performance Overview",
                content_type="metrics",
                data=asdict(performance_metrics),
                description="Key performance indicators for the month"
            ),
            ReportSection(
                title="Monthly Returns Chart",
                content_type="chart",
                data=await self._create_monthly_returns_chart(monthly_data),
                description="Daily returns and cumulative performance"
            ),
            ReportSection(
                title="Strategy Performance",
                content_type="table",
                data=await self._create_strategy_performance_table(monthly_data),
                description="Performance breakdown by trading strategy"
            )
        ])
        
        # Risk analysis
        sections.extend([
            ReportSection(
                title="Risk Metrics",
                content_type="metrics",
                data=asdict(risk_metrics),
                description="Comprehensive risk analysis"
            ),
            ReportSection(
                title="Risk Distribution",
                content_type="chart",
                data=await self._create_risk_distribution_chart(monthly_data),
                description="Risk metrics visualization"
            )
        ])
        
        # Attribution analysis
        sections.append(ReportSection(
            title="Performance Attribution",
            content_type="table",
            data=await self._create_attribution_table(attribution),
            description="Sources of performance by strategy, asset, and time period"
        ))
        
        report = GeneratedReport(
            report_id=f"monthly_{month.replace('-', '')}",
            report_type="monthly_performance",
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary,
            performance_metrics=performance_metrics,
            risk_analysis=risk_metrics,
            attribution_analysis=attribution,
            compliance_metrics=compliance_metrics,
            sections=sections,
            generation_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
        # Export to PDF
        pdf_path = await self._export_report_to_pdf(report)
        report.pdf_path = str(pdf_path)
        report.file_size_bytes = pdf_path.stat().st_size
        
        logger.info(f"Monthly performance report generated: {pdf_path}")
        
        return report
    
    async def risk_report(self, period_days: int = 30) -> GeneratedReport:
        """Generate comprehensive risk analysis report."""
        
        start_time = datetime.utcnow()
        
        period_end = date.today()
        period_start = period_end - timedelta(days=period_days)
        
        logger.info(f"Generating risk report for {period_days} days")
        
        # Fetch data
        risk_data = await self._fetch_period_data(period_start, period_end)
        
        # Calculate comprehensive risk metrics
        risk_metrics = await self._calculate_comprehensive_risk_metrics(risk_data)
        
        # Generate risk-specific sections
        sections = []
        
        # Risk overview
        sections.append(ReportSection(
            title="Risk Overview",
            content_type="metrics",
            data=asdict(risk_metrics),
            description="Comprehensive risk metrics summary"
        ))
        
        # VaR analysis
        var_analysis = await self._perform_var_analysis(risk_data)
        sections.append(ReportSection(
            title="Value at Risk Analysis",
            content_type="table",
            data=var_analysis,
            description="VaR calculations using multiple methodologies"
        ))
        
        # Stress testing results
        stress_results = await self._perform_stress_testing(risk_data)
        sections.append(ReportSection(
            title="Stress Testing Results",
            content_type="table",
            data=stress_results,
            description="Performance under historical stress scenarios"
        ))
        
        # Correlation analysis
        correlation_chart = await self._create_correlation_heatmap(risk_data)
        sections.append(ReportSection(
            title="Correlation Analysis",
            content_type="chart",
            data=correlation_chart,
            description="Strategy and asset correlation matrix"
        ))
        
        # Risk decomposition
        risk_decomp = await self._analyze_risk_decomposition(risk_data)
        sections.append(ReportSection(
            title="Risk Decomposition",
            content_type="chart",
            data=risk_decomp,
            description="Sources of portfolio risk"
        ))
        
        # Create executive summary
        exec_summary = await self._create_risk_executive_summary(risk_metrics)
        
        # Mock performance metrics for risk report
        performance_metrics = ReportMetrics(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, max_drawdown=risk_metrics.value_at_risk_95,
            max_drawdown_duration=0, volatility=0.0, win_rate=0.0,
            profit_factor=0.0, average_win=0.0, average_loss=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            current_streak=0, max_winning_streak=0, max_losing_streak=0
        )
        
        report = GeneratedReport(
            report_id=f"risk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            report_type="risk_analysis",
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary,
            performance_metrics=performance_metrics,
            risk_analysis=risk_metrics,
            attribution_analysis=AttributionAnalysis({}, {}, {}, {}, {}),
            compliance_metrics=ComplianceMetrics(0, 0, 0, 0, 0, 0, 0, []),
            sections=sections,
            generation_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
        # Export to PDF
        pdf_path = await self._export_report_to_pdf(report)
        report.pdf_path = str(pdf_path)
        report.file_size_bytes = pdf_path.stat().st_size
        
        logger.info(f"Risk report generated: {pdf_path}")
        
        return report
    
    async def compliance_report(self, period_days: int = 30) -> GeneratedReport:
        """Generate regulatory compliance reporting."""
        
        start_time = datetime.utcnow()
        
        period_end = date.today()
        period_start = period_end - timedelta(days=period_days)
        
        logger.info(f"Generating compliance report for {period_days} days")
        
        # Fetch compliance data
        compliance_data = await self._fetch_compliance_data(period_start, period_end)
        
        # Calculate compliance metrics
        compliance_metrics = await self._calculate_compliance_metrics(compliance_data)
        
        # Generate compliance sections
        sections = []
        
        # Compliance overview
        sections.append(ReportSection(
            title="Compliance Overview",
            content_type="metrics",
            data=asdict(compliance_metrics),
            description="Summary of regulatory compliance status"
        ))
        
        # Violations summary
        violations_summary = await self._create_violations_summary(compliance_data)
        sections.append(ReportSection(
            title="Violations Summary",
            content_type="table",
            data=violations_summary,
            description="Detailed breakdown of compliance violations"
        ))
        
        # Trading limits monitoring
        limits_monitoring = await self._create_limits_monitoring_table(compliance_data)
        sections.append(ReportSection(
            title="Trading Limits Monitoring",
            content_type="table",
            data=limits_monitoring,
            description="Position and exposure limits tracking"
        ))
        
        # Regulatory alerts
        alerts_summary = await self._create_alerts_summary(compliance_data)
        sections.append(ReportSection(
            title="Regulatory Alerts",
            content_type="table",
            data=alerts_summary,
            description="Summary of regulatory alerts and actions taken"
        ))
        
        # Create executive summary
        exec_summary = await self._create_compliance_executive_summary(compliance_metrics)
        
        # Mock other required metrics
        performance_metrics = ReportMetrics(
            total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, calmar_ratio=0.0, max_drawdown=0.0,
            max_drawdown_duration=0, volatility=0.0, win_rate=0.0,
            profit_factor=0.0, average_win=0.0, average_loss=0.0,
            total_trades=0, winning_trades=0, losing_trades=0,
            current_streak=0, max_winning_streak=0, max_losing_streak=0
        )
        
        risk_metrics = RiskMetrics(
            value_at_risk_95=0.0, value_at_risk_99=0.0, conditional_var_95=0.0,
            conditional_var_99=0.0, beta=0.0, correlation_to_market=0.0,
            tracking_error=0.0, information_ratio=0.0, treynor_ratio=0.0,
            jensen_alpha=0.0
        )
        
        report = GeneratedReport(
            report_id=f"compliance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            report_type="compliance",
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary,
            performance_metrics=performance_metrics,
            risk_analysis=risk_metrics,
            attribution_analysis=AttributionAnalysis({}, {}, {}, {}, {}),
            compliance_metrics=compliance_metrics,
            sections=sections,
            generation_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
        # Export to PDF
        pdf_path = await self._export_report_to_pdf(report)
        report.pdf_path = str(pdf_path)
        report.file_size_bytes = pdf_path.stat().st_size
        
        logger.info(f"Compliance report generated: {pdf_path}")
        
        return report
    
    async def attribution_analysis(self, period_days: int = 30) -> GeneratedReport:
        """Generate performance attribution analysis."""
        
        start_time = datetime.utcnow()
        
        period_end = date.today()
        period_start = period_end - timedelta(days=period_days)
        
        logger.info(f"Generating attribution analysis for {period_days} days")
        
        # Fetch attribution data
        attribution_data = await self._fetch_attribution_data(period_start, period_end)
        
        # Perform comprehensive attribution analysis
        attribution = await self._perform_comprehensive_attribution(attribution_data)
        
        # Generate attribution sections
        sections = []
        
        # Attribution overview
        sections.append(ReportSection(
            title="Attribution Overview",
            content_type="metrics",
            data=asdict(attribution),
            description="Performance attribution breakdown"
        ))
        
        # Strategy attribution
        strategy_attr_chart = await self._create_strategy_attribution_chart(attribution)
        sections.append(ReportSection(
            title="Strategy Attribution",
            content_type="chart",
            data=strategy_attr_chart,
            description="Performance contribution by strategy"
        ))
        
        # Asset attribution
        asset_attr_chart = await self._create_asset_attribution_chart(attribution)
        sections.append(ReportSection(
            title="Asset Attribution",
            content_type="chart",
            data=asset_attr_chart,
            description="Performance contribution by asset class"
        ))
        
        # Time-based attribution
        time_attr_chart = await self._create_time_attribution_chart(attribution)
        sections.append(ReportSection(
            title="Time-based Attribution",
            content_type="chart",
            data=time_attr_chart,
            description="Performance contribution over time"
        ))
        
        # Factor attribution
        factor_attr_table = await self._create_factor_attribution_table(attribution)
        sections.append(ReportSection(
            title="Factor Attribution",
            content_type="table",
            data=factor_attr_table,
            description="Performance attribution to market factors"
        ))
        
        # Create executive summary
        exec_summary = await self._create_attribution_executive_summary(attribution)
        
        # Mock other required metrics
        performance_metrics = ReportMetrics(
            total_return=sum(attribution.strategy_attribution.values()),
            annualized_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
            calmar_ratio=0.0, max_drawdown=0.0, max_drawdown_duration=0,
            volatility=0.0, win_rate=0.0, profit_factor=0.0,
            average_win=0.0, average_loss=0.0, total_trades=0,
            winning_trades=0, losing_trades=0, current_streak=0,
            max_winning_streak=0, max_losing_streak=0
        )
        
        risk_metrics = RiskMetrics(
            value_at_risk_95=0.0, value_at_risk_99=0.0, conditional_var_95=0.0,
            conditional_var_99=0.0, beta=0.0, correlation_to_market=0.0,
            tracking_error=0.0, information_ratio=0.0, treynor_ratio=0.0,
            jensen_alpha=0.0
        )
        
        report = GeneratedReport(
            report_id=f"attribution_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            report_type="attribution_analysis",
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary,
            performance_metrics=performance_metrics,
            risk_analysis=risk_metrics,
            attribution_analysis=attribution,
            compliance_metrics=ComplianceMetrics(0, 0, 0, 0, 0, 0, 0, []),
            sections=sections,
            generation_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
        # Export to PDF
        pdf_path = await self._export_report_to_pdf(report)
        report.pdf_path = str(pdf_path)
        report.file_size_bytes = pdf_path.stat().st_size
        
        logger.info(f"Attribution analysis report generated: {pdf_path}")
        
        return report
    
    async def export_to_pdf(self, report_data: GeneratedReport, custom_template: Optional[str] = None) -> str:
        """Export reports to professional PDF format."""
        
        logger.info(f"Exporting report {report_data.report_id} to PDF")
        
        # Use custom template if provided
        if custom_template and custom_template in self.report_templates:
            template = self.report_templates[custom_template]
        else:
            template = self.report_templates.get(report_data.report_type, self.report_templates['custom'])
        
        # Generate PDF path
        pdf_filename = f"{report_data.report_type}_{report_data.report_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = self.report_dir / pdf_filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=self.page_size,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )
        
        # Build PDF content
        story = []
        
        # Title page
        story.extend(await self._build_title_page(report_data))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(await self._build_executive_summary_section(report_data))
        
        # Report sections
        for section in report_data.sections:
            story.extend(await self._build_section_content(section))
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF generated successfully: {pdf_path}")
        
        return str(pdf_path)
    
    # Private helper methods for data fetching and calculations
    
    async def _fetch_daily_data(self, report_date: date) -> Dict:
        """Fetch daily trading data."""
        # Mock implementation - would fetch real data
        return {
            'date': report_date,
            'total_pnl': np.random.uniform(-1000, 2000),
            'realized_pnl': np.random.uniform(-500, 1000),
            'unrealized_pnl': np.random.uniform(-500, 1000),
            'strategies': {
                f'strategy_{i}': {
                    'pnl': np.random.uniform(-200, 400),
                    'trades': np.random.randint(0, 20),
                    'volume': np.random.uniform(1000, 50000)
                } for i in range(1, 6)
            },
            'positions': [
                {
                    'symbol': f'BTC-USD',
                    'quantity': np.random.uniform(0.1, 2.0),
                    'pnl': np.random.uniform(-100, 200),
                    'unrealized_pnl': np.random.uniform(-50, 100)
                } for _ in range(10)
            ]
        }
    
    async def _fetch_period_data(self, start_date: date, end_date: date) -> Dict:
        """Fetch data for a date range."""
        days = (end_date - start_date).days
        
        # Generate mock time series data
        dates = [start_date + timedelta(days=i) for i in range(days + 1)]
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
        cumulative_returns = np.cumprod(1 + returns) - 1
        
        return {
            'period_start': start_date,
            'period_end': end_date,
            'dates': dates,
            'daily_returns': returns,
            'cumulative_returns': cumulative_returns,
            'total_return': cumulative_returns[-1],
            'volatility': np.std(returns) * np.sqrt(252),
            'strategies': {
                f'strategy_{i}': {
                    'returns': np.random.normal(0.001, 0.015, len(dates)),
                    'total_return': np.random.uniform(-0.1, 0.2),
                    'sharpe_ratio': np.random.uniform(0.5, 2.5),
                    'max_drawdown': np.random.uniform(0.02, 0.15),
                    'trades': np.random.randint(50, 500)
                } for i in range(1, 6)
            }
        }
    
    async def _calculate_daily_metrics(self, daily_data: Dict) -> ReportMetrics:
        """Calculate daily performance metrics."""
        return ReportMetrics(
            total_return=daily_data.get('total_pnl', 0) / 100000,  # Assume 100k portfolio
            annualized_return=0.0,  # Not applicable for daily
            sharpe_ratio=0.0,  # Not applicable for daily
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            volatility=0.0,
            win_rate=0.6,  # Mock
            profit_factor=1.4,
            average_win=150.0,
            average_loss=-80.0,
            total_trades=sum(s['trades'] for s in daily_data['strategies'].values()),
            winning_trades=int(0.6 * sum(s['trades'] for s in daily_data['strategies'].values())),
            losing_trades=int(0.4 * sum(s['trades'] for s in daily_data['strategies'].values())),
            current_streak=3,
            max_winning_streak=7,
            max_losing_streak=3
        )
    
    async def _calculate_period_metrics(self, period_data: Dict) -> ReportMetrics:
        """Calculate comprehensive period metrics."""
        returns = period_data['daily_returns']
        
        return ReportMetrics(
            total_return=period_data['total_return'],
            annualized_return=period_data['total_return'] * (252 / len(returns)),
            sharpe_ratio=np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            sortino_ratio=np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            calmar_ratio=period_data['total_return'] / max(0.01, np.max(np.maximum.accumulate(returns) - returns)),
            max_drawdown=np.max(np.maximum.accumulate(returns) - returns),
            max_drawdown_duration=10,  # Mock
            volatility=period_data['volatility'],
            win_rate=len(returns[returns > 0]) / len(returns),
            profit_factor=np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else float('inf'),
            average_win=np.mean(returns[returns > 0]) if len(returns[returns > 0]) > 0 else 0,
            average_loss=np.mean(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0,
            total_trades=sum(s['trades'] for s in period_data['strategies'].values()),
            winning_trades=int(sum(s['trades'] for s in period_data['strategies'].values()) * 0.6),
            losing_trades=int(sum(s['trades'] for s in period_data['strategies'].values()) * 0.4),
            current_streak=5,
            max_winning_streak=12,
            max_losing_streak=4
        )
    
    async def _calculate_period_risk_metrics(self, period_data: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        returns = period_data['daily_returns']
        
        return RiskMetrics(
            value_at_risk_95=np.percentile(returns, 5),
            value_at_risk_99=np.percentile(returns, 1),
            conditional_var_95=np.mean(returns[returns <= np.percentile(returns, 5)]),
            conditional_var_99=np.mean(returns[returns <= np.percentile(returns, 1)]),
            beta=np.random.uniform(0.8, 1.2),  # Mock
            correlation_to_market=np.random.uniform(0.3, 0.8),
            tracking_error=np.random.uniform(0.02, 0.08),
            information_ratio=np.random.uniform(0.1, 0.5),
            treynor_ratio=np.random.uniform(0.05, 0.15),
            jensen_alpha=np.random.uniform(-0.02, 0.05)
        )
    
    # Additional private methods would continue here with implementations for:
    # - Chart generation methods
    # - Table creation methods  
    # - PDF building methods
    # - Template configurations
    # - Executive summary creation
    # - etc.
    
    def _get_daily_template(self) -> Dict:
        """Get daily report template configuration."""
        return {
            'sections': ['executive_summary', 'pnl_breakdown', 'performance_chart', 'top_performers'],
            'charts': ['performance', 'pnl_breakdown'],
            'metrics_focus': ['daily_pnl', 'trades', 'win_rate']
        }
    
    def _get_monthly_template(self) -> Dict:
        """Get monthly report template configuration."""
        return {
            'sections': ['executive_summary', 'performance_overview', 'risk_analysis', 'attribution'],
            'charts': ['returns', 'risk_distribution', 'strategy_performance'],
            'metrics_focus': ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        }
    
    def _get_quarterly_template(self) -> Dict:
        """Get quarterly report template configuration.""" 
        return self._get_monthly_template()  # Similar to monthly but extended
    
    def _get_annual_template(self) -> Dict:
        """Get annual report template configuration."""
        return self._get_monthly_template()  # Similar to monthly but comprehensive
    
    def _get_custom_template(self) -> Dict:
        """Get custom report template configuration."""
        return {
            'sections': ['executive_summary', 'performance_overview', 'risk_analysis'],
            'charts': ['performance'],
            'metrics_focus': ['total_return', 'sharpe_ratio']
        }
    
    async def _build_title_page(self, report_data: GeneratedReport) -> List:
        """Build PDF title page."""
        elements = []
        
        # Title
        title_text = f"{report_data.report_type.replace('_', ' ').title()} Report"
        elements.append(Paragraph(title_text, self.title_style))
        elements.append(Spacer(1, 30))
        
        # Report details
        details_text = f"""
        <b>Report Period:</b> {report_data.period_start} to {report_data.period_end}<br/>
        <b>Generated:</b> {report_data.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
        <b>Report ID:</b> {report_data.report_id}
        """
        elements.append(Paragraph(details_text, self.styles['Normal']))
        
        return elements
    
    async def _build_executive_summary_section(self, report_data: GeneratedReport) -> List:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.heading_style))
        
        # Summary text
        summary_text = ""
        for key, value in report_data.executive_summary.items():
            if isinstance(value, str):
                summary_text += f"<b>{key.replace('_', ' ').title()}:</b> {value}<br/>"
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        return elements
    
    async def _build_section_content(self, section: ReportSection) -> List:
        """Build content for a report section."""
        elements = []
        
        # Section title
        elements.append(Paragraph(section.title, self.heading_style))
        
        if section.description:
            elements.append(Paragraph(section.description, self.styles['Normal']))
            elements.append(Spacer(1, 10))
        
        # Content based on type
        if section.content_type == 'text':
            elements.append(Paragraph(str(section.data), self.styles['Normal']))
        elif section.content_type == 'table':
            # Create table (simplified)
            elements.append(Paragraph("Table data would be rendered here", self.styles['Normal']))
        elif section.content_type == 'chart':
            elements.append(Paragraph("Chart would be embedded here", self.styles['Normal']))
        elif section.content_type == 'metrics':
            # Format metrics
            metrics_text = ""
            if isinstance(section.data, dict):
                for key, value in section.data.items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"<b>{key.replace('_', ' ').title()}:</b> {value:.4f}<br/>"
            elements.append(Paragraph(metrics_text, self.styles['Normal']))
        
        return elements
    
    # Placeholder methods for complex operations that would be implemented
    
    async def _create_daily_executive_summary(self, daily_data: Dict, metrics: ReportMetrics) -> Dict:
        """Create executive summary for daily report."""
        return {
            'overview': f"Daily P&L of ${daily_data.get('total_pnl', 0):.2f} with {metrics.total_trades} trades executed",
            'highlights': "Strong performance in momentum strategies",
            'concerns': "Increased volatility in crypto markets"
        }
    
    async def _create_monthly_executive_summary(self, monthly_data: Dict, metrics: ReportMetrics) -> Dict:
        """Create executive summary for monthly report."""
        return {
            'overview': f"Monthly return of {metrics.total_return:.2%} with Sharpe ratio of {metrics.sharpe_ratio:.2f}",
            'highlights': "Consistent performance across all strategies",
            'outlook': "Positive momentum expected to continue"
        }
    
    # Additional placeholder methods for chart creation, table building, etc.
    async def _create_pnl_breakdown_table(self, daily_data: Dict) -> Dict: return {}
    async def _create_daily_performance_chart(self, daily_data: Dict) -> Dict: return {}
    async def _analyze_daily_performers(self, daily_data: Dict) -> Dict: return {}
    async def _calculate_daily_risk_metrics(self, daily_data: Dict) -> RiskMetrics: 
        return RiskMetrics(0,0,0,0,0,0,0,0,0,0)
    async def _check_daily_compliance(self, daily_data: Dict) -> ComplianceMetrics:
        return ComplianceMetrics(0,0,0,0,0,0,0,[])
    async def _perform_daily_attribution(self, daily_data: Dict) -> AttributionAnalysis:
        return AttributionAnalysis({},{},{},{},{})
    async def _create_monthly_returns_chart(self, monthly_data: Dict) -> Dict: return {}
    async def _create_strategy_performance_table(self, monthly_data: Dict) -> Dict: return {}
    async def _create_risk_distribution_chart(self, monthly_data: Dict) -> Dict: return {}
    async def _create_attribution_table(self, attribution: AttributionAnalysis) -> Dict: return {}
    async def _perform_period_attribution(self, data: Dict) -> AttributionAnalysis:
        return AttributionAnalysis({},{},{},{},{})
    async def _check_period_compliance(self, data: Dict) -> ComplianceMetrics:
        return ComplianceMetrics(0,0,0,0,0,0,0,[])
    async def _calculate_comprehensive_risk_metrics(self, data: Dict) -> RiskMetrics:
        return RiskMetrics(0,0,0,0,0,0,0,0,0,0)
    async def _perform_var_analysis(self, data: Dict) -> Dict: return {}
    async def _perform_stress_testing(self, data: Dict) -> Dict: return {}
    async def _create_correlation_heatmap(self, data: Dict) -> Dict: return {}
    async def _analyze_risk_decomposition(self, data: Dict) -> Dict: return {}
    async def _create_risk_executive_summary(self, metrics: RiskMetrics) -> Dict: return {}
    async def _fetch_compliance_data(self, start: date, end: date) -> Dict: return {}
    async def _calculate_compliance_metrics(self, data: Dict) -> ComplianceMetrics:
        return ComplianceMetrics(0,0,0,0,0,0,0,[])
    async def _create_violations_summary(self, data: Dict) -> Dict: return {}
    async def _create_limits_monitoring_table(self, data: Dict) -> Dict: return {}
    async def _create_alerts_summary(self, data: Dict) -> Dict: return {}
    async def _create_compliance_executive_summary(self, metrics: ComplianceMetrics) -> Dict: return {}
    async def _fetch_attribution_data(self, start: date, end: date) -> Dict: return {}
    async def _perform_comprehensive_attribution(self, data: Dict) -> AttributionAnalysis:
        return AttributionAnalysis({},{},{},{},{})
    async def _create_strategy_attribution_chart(self, attr: AttributionAnalysis) -> Dict: return {}
    async def _create_asset_attribution_chart(self, attr: AttributionAnalysis) -> Dict: return {}
    async def _create_time_attribution_chart(self, attr: AttributionAnalysis) -> Dict: return {}
    async def _create_factor_attribution_table(self, attr: AttributionAnalysis) -> Dict: return {}
    async def _create_attribution_executive_summary(self, attr: AttributionAnalysis) -> Dict: return {}
    async def _export_report_to_pdf(self, report: GeneratedReport) -> Path:
        """Export report to PDF and return path."""
        return Path(await self.export_to_pdf(report))
    
    def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get report generation status."""
        return {
            'report_id': report_id,
            'status': 'completed',
            'progress': 1.0,
            'generated_at': datetime.utcnow(),
            'file_size': 1024 * 1024  # 1MB mock
        }