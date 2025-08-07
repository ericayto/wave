import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'

export const Settings: React.FC = () => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Settings</CardTitle>
          <CardDescription>Configuration and system settings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <p className="text-gray-400 mb-2">Settings panel coming soon</p>
            <p className="text-sm text-gray-500">Configure API keys, risk limits, and preferences</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}