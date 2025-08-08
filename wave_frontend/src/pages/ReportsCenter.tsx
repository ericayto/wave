import React, { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Progress } from '../components/ui/progress'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Textarea } from '../components/ui/textarea'
import { formatCurrency, formatPercent } from '../lib/utils'
import { 
  FileText,
  Download, 
  Calendar,
  Clock,
  Send,
  Settings,
  BarChart3,
  DollarSign,
  Shield,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Plus,
  Eye,
  Edit,
  Trash2,
  Mail,
  FileSpreadsheet,
  FileImage,
  Archive
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '../components/ui/tabs'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog'

interface GeneratedReport {
  report_id: string
  report_type: string
  title: string
  generated_at: string
  period_start: string
  period_end: string
  file_size_bytes: number
  records_exported: number
  export_formats: string[]
  status: 'generating' | 'completed' | 'failed'
  download_urls: {
    pdf?: string
    csv?: string
    excel?: string
    json?: string
  }
}

interface ScheduledReport {
  schedule_id: string
  name: string
  report_type: string
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly'
  recipients: string[]
  formats: string[]
  last_generated: string
  next_generation: string
  is_active: boolean
  template_config: {
    include_charts: boolean
    include_attribution: boolean
    include_compliance: boolean
  }
}

interface ReportTemplate {
  template_id: string
  name: string
  description: string
  report_type: string
  sections: string[]
  customizable_fields: string[]
  estimated_generation_time: number
}

const mockReports: GeneratedReport[] = [
  {
    report_id: 'daily_20240122',
    report_type: 'daily_pnl',
    title: 'Daily P&L Report - January 22, 2024',
    generated_at: '2024-01-22T18:30:00Z',
    period_start: '2024-01-22',
    period_end: '2024-01-22',
    file_size_bytes: 2457600, // 2.4MB
    records_exported: 156,
    export_formats: ['pdf', 'csv'],
    status: 'completed',
    download_urls: {
      pdf: '/reports/daily_20240122.pdf',
      csv: '/reports/daily_20240122.csv'
    }
  },
  {
    report_id: 'monthly_202401',
    report_type: 'monthly_performance',
    title: 'Monthly Performance Report - January 2024',
    generated_at: '2024-01-21T15:45:00Z',
    period_start: '2024-01-01',
    period_end: '2024-01-21',
    file_size_bytes: 5242880, // 5MB
    records_exported: 2340,
    export_formats: ['pdf', 'excel', 'json'],
    status: 'completed',
    download_urls: {
      pdf: '/reports/monthly_202401.pdf',
      excel: '/reports/monthly_202401.xlsx',
      json: '/reports/monthly_202401.json'
    }
  },
  {
    report_id: 'risk_20240122',
    report_type: 'risk_analysis',
    title: 'Risk Analysis Report - January 22, 2024',
    generated_at: '2024-01-22T12:15:00Z',
    period_start: '2024-01-01',
    period_end: '2024-01-22',
    file_size_bytes: 1843200, // 1.8MB
    records_exported: 890,
    export_formats: ['pdf'],
    status: 'generating',
    download_urls: {}
  }
]

const mockScheduledReports: ScheduledReport[] = [
  {
    schedule_id: 'sched_001',
    name: 'Daily Performance Summary',
    report_type: 'daily_pnl',
    frequency: 'daily',
    recipients: ['trader@example.com', 'manager@example.com'],
    formats: ['pdf', 'csv'],
    last_generated: '2024-01-22T08:00:00Z',
    next_generation: '2024-01-23T08:00:00Z',
    is_active: true,
    template_config: {
      include_charts: true,
      include_attribution: false,
      include_compliance: true
    }
  },
  {
    schedule_id: 'sched_002',
    name: 'Weekly Risk Report',
    report_type: 'risk_analysis',
    frequency: 'weekly',
    recipients: ['risk@example.com'],
    formats: ['pdf'],
    last_generated: '2024-01-15T09:00:00Z',
    next_generation: '2024-01-22T09:00:00Z',
    is_active: true,
    template_config: {
      include_charts: true,
      include_attribution: false,
      include_compliance: true
    }
  }
]

const mockReportTemplates: ReportTemplate[] = [
  {
    template_id: 'daily_pnl_template',
    name: 'Daily P&L Report',
    description: 'Comprehensive daily profit and loss analysis with trading activity breakdown',
    report_type: 'daily_pnl',
    sections: ['executive_summary', 'pnl_breakdown', 'performance_chart', 'top_performers'],
    customizable_fields: ['include_intraday_charts', 'show_position_details', 'include_risk_metrics'],
    estimated_generation_time: 45
  },
  {
    template_id: 'monthly_performance_template',
    name: 'Monthly Performance Report',
    description: 'Complete monthly performance analysis with attribution and risk metrics',
    report_type: 'monthly_performance',
    sections: ['performance_overview', 'strategy_breakdown', 'risk_analysis', 'attribution'],
    customizable_fields: ['include_benchmarking', 'show_correlation_matrix', 'include_optimization_suggestions'],
    estimated_generation_time: 120
  },
  {
    template_id: 'risk_analysis_template',
    name: 'Risk Analysis Report',
    description: 'Comprehensive risk assessment with VaR, stress testing, and correlation analysis',
    report_type: 'risk_analysis',
    sections: ['risk_overview', 'var_analysis', 'stress_testing', 'correlation_matrix'],
    customizable_fields: ['include_scenario_analysis', 'show_tail_risk_metrics', 'include_liquidity_analysis'],
    estimated_generation_time: 90
  },
  {
    template_id: 'compliance_template',
    name: 'Compliance Report',
    description: 'Regulatory compliance monitoring and violation tracking',
    report_type: 'compliance',
    sections: ['compliance_overview', 'violations_summary', 'limits_monitoring', 'regulatory_alerts'],
    customizable_fields: ['include_audit_trail', 'show_limit_utilization', 'include_policy_exceptions'],
    estimated_generation_time: 60
  }
]

const fetchReports = async (): Promise<GeneratedReport[]> => {
  await new Promise(resolve => setTimeout(resolve, 500))
  return mockReports
}

const fetchScheduledReports = async (): Promise<ScheduledReport[]> => {
  await new Promise(resolve => setTimeout(resolve, 300))
  return mockScheduledReports
}

const fetchReportTemplates = async (): Promise<ReportTemplate[]> => {
  await new Promise(resolve => setTimeout(resolve, 200))
  return mockReportTemplates
}

export const ReportsCenter: React.FC = () => {
  const [isGenerateDialogOpen, setIsGenerateDialogOpen] = useState(false)
  const [isScheduleDialogOpen, setIsScheduleDialogOpen] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [reportConfig, setReportConfig] = useState({
    period_start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    period_end: new Date().toISOString().split('T')[0],
    formats: ['pdf'],
    include_charts: true,
    include_attribution: true
  })

  const { data: reports, isLoading: reportsLoading, refetch: refetchReports } = useQuery({
    queryKey: ['reports'],
    queryFn: fetchReports,
    refetchInterval: 10000, // Refetch every 10 seconds to update generation status
  })

  const { data: scheduledReports, isLoading: scheduledLoading } = useQuery({
    queryKey: ['scheduled-reports'],
    queryFn: fetchScheduledReports,
  })

  const { data: reportTemplates, isLoading: templatesLoading } = useQuery({
    queryKey: ['report-templates'],
    queryFn: fetchReportTemplates,
  })

  const generateReportMutation = useMutation({
    mutationFn: async (config: any) => {
      await new Promise(resolve => setTimeout(resolve, 2000))
      return { 
        report_id: 'report_' + Date.now(),
        status: 'generating',
        estimated_completion: new Date(Date.now() + 2 * 60 * 1000).toISOString()
      }
    },
    onSuccess: () => {
      setIsGenerateDialogOpen(false)
      refetchReports()
    },
  })

  const createScheduleMutation = useMutation({
    mutationFn: async (scheduleConfig: any) => {
      await new Promise(resolve => setTimeout(resolve, 1000))
      return { schedule_id: 'sched_' + Date.now() }
    },
    onSuccess: () => {
      setIsScheduleDialogOpen(false)
    },
  })

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'generating': return 'text-blue-400 bg-blue-400/10'
      case 'completed': return 'text-green-400 bg-green-400/10'
      case 'failed': return 'text-red-400 bg-red-400/10'
      default: return 'text-gray-400 bg-gray-400/10'
    }
  }

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'pdf': return <FileText className="w-4 h-4" />
      case 'csv': return <FileSpreadsheet className="w-4 h-4" />
      case 'excel': return <FileSpreadsheet className="w-4 h-4" />
      case 'json': return <Archive className="w-4 h-4" />
      default: return <FileText className="w-4 h-4" />
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  const formatFrequency = (frequency: string) => {
    return frequency.charAt(0).toUpperCase() + frequency.slice(1)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Reports Center</h1>
          <p className="text-gray-400 mt-2">Generate, schedule, and manage professional trading reports</p>
        </div>
        <div className="flex items-center space-x-4">
          <Dialog open={isScheduleDialogOpen} onOpenChange={setIsScheduleDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <Calendar className="w-4 h-4 mr-2" />
                Schedule Report
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[600px]">
              <DialogHeader>
                <DialogTitle>Schedule Automated Report</DialogTitle>
                <DialogDescription>
                  Configure automated report generation and distribution
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Report Name</Label>
                    <Input placeholder="e.g., Daily Trading Summary" className="mt-1" />
                  </div>
                  <div>
                    <Label>Report Type</Label>
                    <Select>
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="daily_pnl">Daily P&L</SelectItem>
                        <SelectItem value="monthly_performance">Monthly Performance</SelectItem>
                        <SelectItem value="risk_analysis">Risk Analysis</SelectItem>
                        <SelectItem value="compliance">Compliance</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Frequency</Label>
                    <Select>
                      <SelectTrigger className="mt-1">
                        <SelectValue placeholder="Select frequency" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="daily">Daily</SelectItem>
                        <SelectItem value="weekly">Weekly</SelectItem>
                        <SelectItem value="monthly">Monthly</SelectItem>
                        <SelectItem value="quarterly">Quarterly</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Time</Label>
                    <Input type="time" defaultValue="08:00" className="mt-1" />
                  </div>
                </div>
                
                <div>
                  <Label>Recipients (comma-separated)</Label>
                  <Textarea 
                    placeholder="email1@example.com, email2@example.com"
                    className="mt-1"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button 
                  onClick={() => createScheduleMutation.mutate({})}
                  disabled={createScheduleMutation.isPending}
                >
                  {createScheduleMutation.isPending ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    'Create Schedule'
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
          
          <Dialog open={isGenerateDialogOpen} onOpenChange={setIsGenerateDialogOpen}>
            <DialogTrigger asChild>
              <Button className="bg-gradient-to-r from-ocean-500 to-wave-500">
                <FileText className="w-4 h-4 mr-2" />
                Generate Report
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-[600px]">
              <DialogHeader>
                <DialogTitle>Generate New Report</DialogTitle>
                <DialogDescription>
                  Configure and generate a custom trading report
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label>Report Template</Label>
                  <Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select report template" />
                    </SelectTrigger>
                    <SelectContent>
                      {reportTemplates?.map((template) => (
                        <SelectItem key={template.template_id} value={template.template_id}>
                          {template.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  {selectedTemplate && reportTemplates && (
                    <p className="text-xs text-gray-400 mt-1">
                      {reportTemplates.find(t => t.template_id === selectedTemplate)?.description}
                    </p>
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Start Date</Label>
                    <Input 
                      type="date" 
                      value={reportConfig.period_start}
                      onChange={(e) => setReportConfig(prev => ({...prev, period_start: e.target.value}))}
                      className="mt-1" 
                    />
                  </div>
                  <div>
                    <Label>End Date</Label>
                    <Input 
                      type="date" 
                      value={reportConfig.period_end}
                      onChange={(e) => setReportConfig(prev => ({...prev, period_end: e.target.value}))}
                      className="mt-1" 
                    />
                  </div>
                </div>
                
                <div>
                  <Label>Export Formats</Label>
                  <div className="mt-2 space-y-2">
                    {['pdf', 'csv', 'excel', 'json'].map((format) => (
                      <label key={format} className="flex items-center space-x-2">
                        <input 
                          type="checkbox" 
                          checked={reportConfig.formats.includes(format)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setReportConfig(prev => ({
                                ...prev, 
                                formats: [...prev.formats, format]
                              }))
                            } else {
                              setReportConfig(prev => ({
                                ...prev,
                                formats: prev.formats.filter(f => f !== format)
                              }))
                            }
                          }}
                          className="rounded"
                        />
                        <span className="text-sm text-white uppercase">{format}</span>
                      </label>
                    ))}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input 
                      type="checkbox" 
                      checked={reportConfig.include_charts}
                      onChange={(e) => setReportConfig(prev => ({...prev, include_charts: e.target.checked}))}
                      className="rounded"
                    />
                    <span className="text-sm text-white">Include Charts</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input 
                      type="checkbox" 
                      checked={reportConfig.include_attribution}
                      onChange={(e) => setReportConfig(prev => ({...prev, include_attribution: e.target.checked}))}
                      className="rounded"
                    />
                    <span className="text-sm text-white">Include Attribution Analysis</span>
                  </label>
                </div>
              </div>
              <DialogFooter>
                <Button 
                  onClick={() => generateReportMutation.mutate({...reportConfig, template: selectedTemplate})}
                  disabled={!selectedTemplate || generateReportMutation.isPending}
                >
                  {generateReportMutation.isPending ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    'Generate Report'
                  )}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Tabs defaultValue="reports" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="reports">Generated Reports</TabsTrigger>
          <TabsTrigger value="scheduled">Scheduled Reports</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="exports">Data Exports</TabsTrigger>
        </TabsList>

        <TabsContent value="reports" className="space-y-6">
          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Total Reports
                </CardTitle>
                <FileText className="h-4 w-4 text-ocean-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">
                  {reports?.length || 0}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  This month
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Generating
                </CardTitle>
                <RefreshCw className="h-4 w-4 text-blue-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-400">
                  {reports?.filter(r => r.status === 'generating').length || 0}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  In progress
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Total Size
                </CardTitle>
                <Archive className="h-4 w-4 text-wave-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">
                  {reports ? formatFileSize(reports.reduce((sum, r) => sum + r.file_size_bytes, 0)) : '0 B'}
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Storage used
                </p>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-gray-400">
                  Downloads
                </CardTitle>
                <Download className="h-4 w-4 text-green-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-400">
                  247
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  This month
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Reports List */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Reports</CardTitle>
              <CardDescription>Generated reports and their status</CardDescription>
            </CardHeader>
            <CardContent>
              {reportsLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} className="shimmer h-16 rounded"></div>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {reports?.map((report) => (
                    <div key={report.report_id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-ocean-400/10 rounded-lg flex items-center justify-center">
                          <FileText className="w-5 h-5 text-ocean-400" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2">
                            <h4 className="font-medium text-white">{report.title}</h4>
                            <Badge className={getStatusColor(report.status)}>
                              {report.status}
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-4 mt-1 text-sm text-gray-400">
                            <span>{new Date(report.generated_at).toLocaleDateString()}</span>
                            <span>{formatFileSize(report.file_size_bytes)}</span>
                            <span>{report.records_exported.toLocaleString()} records</span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        {report.status === 'generating' ? (
                          <div className="flex items-center space-x-2">
                            <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
                            <Progress value={65} className="w-20 h-2" />
                          </div>
                        ) : report.status === 'completed' ? (
                          <div className="flex items-center space-x-1">
                            {Object.entries(report.download_urls).map(([format, url]) => (
                              <Button key={format} size="sm" variant="outline" asChild>
                                <a href={url} download>
                                  {getFormatIcon(format)}
                                </a>
                              </Button>
                            ))}
                          </div>
                        ) : (
                          <Badge className="text-red-400 bg-red-400/10">
                            Failed
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                  
                  {(!reports || reports.length === 0) && (
                    <div className="text-center py-8 text-gray-500">
                      <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No reports generated yet</p>
                      <p className="text-xs mt-1">Create your first report using the generate button above</p>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="scheduled" className="space-y-6">
          {/* Scheduled Reports Management */}
          <Card>
            <CardHeader>
              <CardTitle>Scheduled Reports</CardTitle>
              <CardDescription>Automated report generation and distribution</CardDescription>
            </CardHeader>
            <CardContent>
              {scheduledLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <div key={i} className="shimmer h-20 rounded"></div>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  {scheduledReports?.map((schedule) => (
                    <div key={schedule.schedule_id} className="p-4 bg-white/5 rounded-lg">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className={`w-2 h-2 rounded-full ${schedule.is_active ? 'bg-green-400' : 'bg-gray-400'}`}></div>
                          <h4 className="font-medium text-white">{schedule.name}</h4>
                          <Badge className="text-ocean-400 bg-ocean-400/10">
                            {formatFrequency(schedule.frequency)}
                          </Badge>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Button size="sm" variant="outline">
                            <Edit className="w-3 h-3" />
                          </Button>
                          <Button size="sm" variant="outline">
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <div className="text-gray-400">Recipients</div>
                          <div className="text-white">{schedule.recipients.length} recipients</div>
                          <div className="text-xs text-gray-500 truncate">
                            {schedule.recipients.join(', ')}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-400">Formats</div>
                          <div className="flex items-center space-x-1 mt-1">
                            {schedule.formats.map((format) => (
                              <Badge key={format} variant="outline" className="text-xs">
                                {format.toUpperCase()}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-400">Next Generation</div>
                          <div className="text-white">
                            {new Date(schedule.next_generation).toLocaleDateString()}
                          </div>
                          <div className="text-xs text-gray-400">
                            {new Date(schedule.next_generation).toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {(!scheduledReports || scheduledReports.length === 0) && (
                    <div className="text-center py-8 text-gray-500">
                      <Calendar className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No scheduled reports</p>
                      <p className="text-xs mt-1">Set up automated report generation</p>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="templates" className="space-y-6">
          {/* Report Templates */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {reportTemplates?.map((template) => (
              <Card key={template.template_id} className="glow-hover">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{template.name}</span>
                    <Badge className="text-ocean-400 bg-ocean-400/10">
                      {template.report_type.replace('_', ' ').toUpperCase()}
                    </Badge>
                  </CardTitle>
                  <CardDescription>{template.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h5 className="font-medium text-white mb-2">Sections ({template.sections.length})</h5>
                    <div className="flex flex-wrap gap-1">
                      {template.sections.map((section) => (
                        <Badge key={section} variant="outline" className="text-xs">
                          {section.replace('_', ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h5 className="font-medium text-white mb-2">Customizable Options</h5>
                    <div className="space-y-1 text-sm">
                      {template.customizable_fields.map((field) => (
                        <div key={field} className="flex items-center space-x-2">
                          <CheckCircle className="w-3 h-3 text-green-400" />
                          <span className="text-gray-300">{field.replace('_', ' ')}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="pt-4 border-t border-white/10 flex items-center justify-between">
                    <div className="flex items-center space-x-2 text-sm text-gray-400">
                      <Clock className="w-4 h-4" />
                      <span>~{template.estimated_generation_time}s</span>
                    </div>
                    <Button size="sm" onClick={() => {
                      setSelectedTemplate(template.template_id)
                      setIsGenerateDialogOpen(true)
                    }}>
                      Use Template
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Custom Template Creation */}
          <Card className="border-dashed border-gray-600">
            <CardContent className="p-6 text-center">
              <Plus className="w-8 h-8 text-gray-500 mx-auto mb-2" />
              <h4 className="font-medium text-white mb-2">Create Custom Template</h4>
              <p className="text-sm text-gray-400 mb-4">
                Design your own report template with custom sections and formatting
              </p>
              <Button variant="outline">
                <Plus className="w-4 h-4 mr-2" />
                Create Template
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="exports" className="space-y-6">
          {/* Data Export Options */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="glow-hover">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileSpreadsheet className="w-5 h-5 text-green-400" />
                  <span>Tax Reporting</span>
                </CardTitle>
                <CardDescription>Export data for tax preparation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-300">
                  Generate tax-compliant reports including Form 8949 format, cost basis calculations, and gain/loss summaries.
                </p>
                <Button className="w-full" size="sm">
                  <DollarSign className="w-4 h-4 mr-2" />
                  Export Tax Data
                </Button>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Archive className="w-5 h-5 text-blue-400" />
                  <span>Portfolio Tracker</span>
                </CardTitle>
                <CardDescription>Integration with external tools</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-300">
                  Export data for Personal Capital, Mint, QuickBooks, and other portfolio tracking platforms.
                </p>
                <Button className="w-full" size="sm">
                  <Send className="w-4 h-4 mr-2" />
                  Export for Integration
                </Button>
              </CardContent>
            </Card>

            <Card className="glow-hover">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-ocean-400" />
                  <span>Raw Data Export</span>
                </CardTitle>
                <CardDescription>Bulk data export options</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-sm text-gray-300">
                  Export raw trading data, positions, orders, and performance metrics in various formats.
                </p>
                <Button className="w-full" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Export Raw Data
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Export History */}
          <Card>
            <CardHeader>
              <CardTitle>Export History</CardTitle>
              <CardDescription>Recent data export activities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { type: 'Tax Export 2023', date: '2024-01-20', size: '1.2 MB', format: 'CSV' },
                  { type: 'Portfolio Sync', date: '2024-01-18', size: '856 KB', format: 'JSON' },
                  { type: 'Trading Data', date: '2024-01-15', size: '3.4 MB', format: 'Excel' }
                ].map((export, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-wave-400/10 rounded-lg flex items-center justify-center">
                        {getFormatIcon(export.format.toLowerCase())}
                      </div>
                      <div>
                        <div className="text-sm font-medium text-white">{export.type}</div>
                        <div className="text-xs text-gray-400">
                          {export.date} • {export.size} • {export.format}
                        </div>
                      </div>
                    </div>
                    <Button size="sm" variant="outline">
                      <Download className="w-3 h-3" />
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}