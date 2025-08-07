import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'

export const Strategies: React.FC = () => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Trading Strategies</CardTitle>
          <CardDescription>Manage and configure your trading strategies</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <p className="text-gray-400 mb-2">Strategy management coming soon</p>
            <p className="text-sm text-gray-500">Create and deploy automated trading strategies</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}